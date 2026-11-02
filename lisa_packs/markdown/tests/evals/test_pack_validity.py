"""Pack-validation eval: the markdown pack stays structurally healthy.

Runs under the framework's ``eval`` marker (deterministic, no model calls).
Validates the pack from source via ``packstest.catalog.validate_pack``:
distribution metadata, entry points, tool YAML contract, and in-process
importability of every implementation module. First eval any pack edit
should run — it fails fast and names the exact file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packstest.catalog import validate_pack

PACK_PATH = Path(__file__).resolve().parents[4]  # .../lisa-packs/markdown

EXPECTED_TOOLS = {"markdown_generate"}


@pytest.mark.eval
def test_markdown_pack_validates_from_source() -> None:
    """Entry points load, every tool YAML parses, modules import in-process."""

    report = validate_pack(PACK_PATH)
    assert report.ok, report.summary()


@pytest.mark.eval
def test_markdown_pack_tool_catalog_complete() -> None:
    """Every expected tool is discovered with the markdown provider id."""

    report = validate_pack(PACK_PATH)
    missing = EXPECTED_TOOLS - set(report.tools)
    unexpected = set(report.tools) - EXPECTED_TOOLS
    assert not missing, f"tools missing from catalog: {sorted(missing)}"
    assert not unexpected, f"unexpected catalog tools: {sorted(unexpected)}"
    for tool_id, tool_dict in sorted(report.tools.items()):
        assert tool_dict["tool_provider_id"] == "lisa-pack-markdown", tool_id
        assert tool_dict["pack_id"] == "lisa-pack-markdown", tool_id
        assert tool_dict["effect"] == "read", f"{tool_id}: unexpected effect"


@pytest.mark.eval
def test_markdown_pack_entry_points_load() -> None:
    """The content-root entry point resolves from source."""

    report = validate_pack(PACK_PATH)
    assert "lisa.extensions.content_roots" in report.entry_points
    for group, entries in report.entry_points.items():
        for name, value in entries.items():
            assert value is not None, f"{group}:{name} failed to load"