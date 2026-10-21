"""Catalog discovery test for the markdown pack."""

from __future__ import annotations

import json

from sdk.content import (
    build_operation_catalog_json,
    build_tool_catalog_json,
)

from lisa_packs import markdown


def test_forge_document_discovered() -> None:
    root = markdown.get_content_root() / "operations"
    catalog = json.loads(build_operation_catalog_json([root]))
    assert "markdown_forge_document" in catalog
    op = catalog["markdown_forge_document"]
    assert op["description"]


def test_forge_document_tool_ids_declared() -> None:
    root = markdown.get_content_root() / "operations"
    catalog = json.loads(build_operation_catalog_json([root]))
    tool_ids = set(catalog["markdown_forge_document"]["tool_ids"])
    assert {
        "markdown_generate",
        "write_file",
        "read_file",
        "memory.search",
    } <= tool_ids


def test_consolidate_operation_discovered() -> None:
    root = markdown.get_content_root() / "operations"
    catalog = json.loads(build_operation_catalog_json([root]))
    assert "markdown_consolidate_sources" in catalog
    op = catalog["markdown_consolidate_sources"]
    assert op["description"]
    tool_ids = set(op["tool_ids"])
    assert {
        "markdown_generate",
        "write_file",
        "read_file",
        "list_files",
        "memory.search",
        "okf_emit_concept",
        "okf_log",
    } <= tool_ids


def test_outline_markdown_tool_discovered() -> None:
    root = markdown.get_content_root() / "tools"
    catalog = json.loads(build_tool_catalog_json([root]))
    assert "markdown_generate" in catalog
    tool = catalog["markdown_generate"]
    assert tool["description"]
    assert tool["pack_id"] == "lisa-pack-markdown"


def test_render_turn_discovered() -> None:
    root = markdown.get_content_root() / "operations"
    catalog = json.loads(build_operation_catalog_json([root]))
    assert "markdown_render_turn" in catalog
    op = catalog["markdown_render_turn"]
    assert op["description"]
