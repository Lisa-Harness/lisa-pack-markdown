"""Shared long-term memory recall for markdown pack operations.

Both ``markdown_forge_document`` (single-doc forge) and
``markdown_consolidate_sources`` (multi-source consolidation) ground their
document in recalled long-term memory: the operation issues a
``memory.search`` call through the host's typed tool seam and injects the
returned records into the forge prompt.

The two operations ran byte-identical recall logic before this module
existed — the same ``memory.search`` tool id, the same scoped parameters,
the same limit clamp, and the same three output shapes (a JSON list, a
JSON object with a ``results`` list, or raw text). This module owns that
logic once so the two operations cannot drift apart; each keeps its own
operation-local ``_recall_memory`` wrapper (and its own error message) so
the operation-level contract stays visible at the call site.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from kernel.models import OperationExecutionContext, ToolExecutionContext

logger = logging.getLogger(__name__)

__all__ = ["MEMORY_SEARCH_TOOL_ID", "recall_memory", "render_memory_lines"]

#: Tool id for the memory backend's search alias, routed through
#: ``ctx.host.execute_tool`` (the host resolves the alias to the configured
#: memory backend).
MEMORY_SEARCH_TOOL_ID = "memory.search"

#: Upper bound on the number of records a single recall may request. The
#: memory backend caps its own limit; clamping here keeps a mis-configured
#: ``limit`` from asking for an unbounded page.
_MAX_RECALL_LIMIT = 20


def _render_items(items: list[Any], *, allow_bare_strings: bool) -> str:
    """Render memory records as a newline-separated bullet list.

    A top-level list may carry either dicts (with a ``content`` key) or
    bare strings; a ``results`` list carries dicts only. ``allow_bare_strings``
    selects between the two shapes.
    """
    lines: list[str] = []
    for item in items:
        if isinstance(item, dict):
            content = str(item.get("content") or "").strip()
            if content:
                lines.append(f"- {content}")
        elif allow_bare_strings and isinstance(item, str) and item.strip():
            lines.append(f"- {item.strip()}")
    return "\n".join(lines)


def render_memory_lines(data: Any) -> str | None:
    """Render a decoded memory-search payload as a context block.

    Returns ``None`` when the payload carries no recognized list shape, so
    the caller can fall back to the raw text.
    """
    if isinstance(data, list):
        return _render_items(data, allow_bare_strings=True)
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return _render_items(data["results"], allow_bare_strings=False)
    return None


async def recall_memory(
    ctx: OperationExecutionContext,
    query: str,
    *,
    limit: int,
) -> str:
    """Recall relevant memories via the ``memory.search`` MCP alias.

    Returns a newline-formatted context block (possibly empty). The raw
    tool output is returned verbatim when it is not JSON, and an empty
    string is returned when the tool reports an error (the recall is
    advisory — a memory-backend failure must not fail the forge).

    Raises ``RuntimeError`` only when the tool executor itself is missing.
    """
    if ctx.host is None:
        raise RuntimeError("markdown requires OperationHost to call memory/storage tools")

    recall_limit = max(1, min(int(limit), _MAX_RECALL_LIMIT))
    tool_result = await ctx.host.execute_tool(
        MEMORY_SEARCH_TOOL_ID,
        ToolExecutionContext(
            tool_id=MEMORY_SEARCH_TOOL_ID,
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
    rendered = render_memory_lines(data)
    if rendered is None:
        return output_text.strip()
    return rendered
