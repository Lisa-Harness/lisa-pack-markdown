"""Tests for the advisory markdown linter."""

from __future__ import annotations

import logging

import pytest

from lisa_packs.markdown.operations._lint import lint_markdown

pytest.importorskip("pymarkdown")


def test_scan_leaves_the_host_root_logger_untouched() -> None:
    """``pymarkdown`` reconfigures root logging on every scan.

    Regression: after the first forge
    lint the root level was WARNING and a stdout handler was attached, so
    every later INFO record was dropped and warnings leaked onto stdout.
    """
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    saved_level, saved_handlers = root.level, list(root.handlers)
    root.setLevel(logging.INFO)
    root.addHandler(sentinel)
    try:
        issues = lint_markdown("# Title\n\nNo final newline.")

        assert root.level == logging.INFO
        assert root.handlers == [*saved_handlers, sentinel]
        assert any(issue.rule_id == "md047" for issue in issues)
    finally:
        root.removeHandler(sentinel)
        root.setLevel(saved_level)


def test_scan_without_root_handlers_adds_none() -> None:
    root = logging.getLogger()
    saved_level, saved_handlers = root.level, list(root.handlers)
    for handler in saved_handlers:
        root.removeHandler(handler)
    try:
        lint_markdown("# Title\n\ntext\n")
        assert root.handlers == []
    finally:
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)
