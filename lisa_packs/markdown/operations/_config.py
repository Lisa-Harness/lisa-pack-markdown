"""Shared config-resolution helpers for markdown pack operations.

Both ``markdown_forge_document`` and ``markdown_consolidate_sources`` read
session-manifest configuration from ``ctx.pack_config`` with module-constant
fallbacks. The ``_cfg_*`` coercion helpers live here so neither operation
re-implements them.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["cfg_bool", "cfg_float", "cfg_int", "cfg_str"]


def cfg_str(pack_config: Any, key: str, default: str) -> str:
    """Read a string from ``ctx.pack_config`` with a module-constant fallback.

    Same contract as :func:`cfg_int` but for free-form string values (e.g.
    a CSV of lint rule identifiers). Coerces non-str values to their str()
    form; ``None``/``""`` → default.
    """
    if pack_config is None:
        return default
    raw = pack_config.get(key)
    if raw is None:
        return default
    text = str(raw)
    if not text.strip():
        return default
    return text


def cfg_int(pack_config: Any, key: str, default: int) -> int:
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
        logger.warning(
            "markdown: pack_config[%r]=%r is not an int; using default %d", key, raw, default
        )
        return default


def cfg_float(pack_config: Any, key: str, default: float) -> float:
    """Read a float from ``ctx.pack_config`` with a module-constant fallback.

    Same contract as :func:`cfg_int` but for ratio/percentage thresholds.
    """
    if pack_config is None:
        return default
    raw = pack_config.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "markdown: pack_config[%r]=%r is not a float; using default %s", key, raw, default
        )
        return default


def cfg_bool(pack_config: Any, key: str, default: bool) -> bool:
    """Read a bool from ``ctx.pack_config`` with a module-constant fallback.

    Accepts native bools and the string forms ``true``/``false``/``1``/``0``
    (case-insensitive). Coerces anything else to the default rather than
    raising.
    """
    if pack_config is None:
        return default
    raw = pack_config.get(key)
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off", ""):
        return False
    logger.warning(
        "markdown: pack_config[%r]=%r is not a bool; using default %s", key, raw, default
    )
    return default
