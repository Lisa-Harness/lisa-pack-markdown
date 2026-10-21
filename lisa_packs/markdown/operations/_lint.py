"""Advisory markdown linting for the markdown pack operations.

Complements :func:`lisa_packs.markdown.operations.forge_document.forge_document._format_markdown`
(which *fixes* via ``mdformat``) with a *reporter*: this module runs
``pymarkdownlnt`` (the ``pymarkdown`` package) against a document and
returns a list of structural rule violations without modifying the text.

The contract is **advisory, never a hard reject**. The outline gate
(``markdown_generate`` ≥1 heading) stays the only hard gate; lint issues
feed into at most one repair pass and, if any remain, are persisted as a
non-rendering HTML comment so the user knows the artifact has warnings.

Rule scope is deliberately narrow: only structural rules that affect
rendering correctness or document integrity are surfaced. Prose-style
rules (line length, emphasis-as-heading, inline HTML) are excluded
because they over-fire on LLM-authored content and would create noise
without value. The allowed set is overridable via the ``forge_lint_rules``
``pack_config`` key (CSV of rule IDs).
"""

from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["STRUCTURAL_RULES", "LintIssue", "format_lint_issues", "lint_markdown"]

# Curated structural rule subset. These rules flag issues that affect
# rendering correctness or document integrity, not stylistic preference.
# MD001  — heading levels increment by one at a time
# MD025  — exactly one top-level (H1) heading
# MD041  — first line is a top-level heading
# MD047  — file ends with a single newline
# MD009  — trailing whitespace
# MD012  — multiple consecutive blank lines
# MD042  — empty links
# MD051  — link fragments reference defined headings
#
# Intentionally excluded (prose/style rules that over-fire on LLM output):
#   MD013 (line length), MD033 (inline HTML — we emit HTML comments),
#   MD036 (emphasis as heading), MD024 (duplicate headings — often
#   legitimate in reference docs), MD031 (spaces around spans).
STRUCTURAL_RULES: tuple[str, ...] = (
    "md001",
    "md025",
    "md041",
    "md047",
    "md009",
    "md012",
    "md042",
    "md051",
)

# Prose/style rules to disable even if the default configuration enables
# them. Disabling is idempotent and stable across pymarkdownlnt versions;
# enumerating *all* rule ids to do an enable-only approach is not.
_NOISY_RULES: tuple[str, ...] = (
    "md013",  # line length
    "md033",  # inline HTML (we emit HTML comments intentionally)
    "md036",  # emphasis as heading
    "md024",  # multiple headings with the same content
    "md031",  # spaces around emphasis/link spans
    "md022",  # headings should be surrounded by blank lines (mdformat fixes)
    "md032",  # lists should be surrounded by blank lines (mdformat fixes)
    "md007",  # unordered list indentation
    "md005",  # list indentation consistency
    "md038",  # spaces inside code span elements
    "md037",  # spaces inside emphasis markers
)


@dataclass(frozen=True)
class LintIssue:
    """A single lint violation."""

    rule_id: str
    line_number: int
    description: str


def _parse_rule_list(raw: str | None) -> list[str]:
    """Parse a CSV string of rule ids into a normalized list.

    Accepts ``"md001, MD025, md047"`` (whitespace-insensitive,
    case-insensitive). Returns ``[]`` when ``raw`` is falsy.
    """
    if not raw:
        return []
    parts = re.split(r"[\s,;]+", raw.strip())
    return [p.lower() for p in parts if p]


def _enabled_rules(override: str | None) -> tuple[str, ...]:
    """Resolve the effective rule set: override CSV or the curated default."""
    parsed = _parse_rule_list(override)
    if parsed:
        return tuple(parsed)
    return STRUCTURAL_RULES


def lint_markdown(
    document: str,
    *,
    rules_csv: str | None = None,
) -> list[LintIssue]:
    """Run ``pymarkdownlnt`` against ``document`` and return violations.

    Returns an empty list when ``pymarkdownlnt`` is not installed (soft
    dependency — same precedent as :func:`_format_markdown`) or when the
    scan raises. Never raises; the caller treats lint as advisory.

    ``rules_csv`` optionally overrides the curated :data:`STRUCTURAL_RULES`
    set with a CSV of rule ids (e.g. ``"md001,md041"``).
    """
    try:
        from pymarkdown.api import (
            PyMarkdownApi,
            PyMarkdownApiException,
        )
    except ImportError:
        logger.debug("pymarkdownlnt not installed; skipping markdown lint")
        return []
    if not document.strip():
        return []

    api = PyMarkdownApi()
    api.enable_continue_on_error()

    # Configure the rule set: disable the noisy prose rules, then narrow
    # to the enabled subset by disabling anything not in the allow-set.
    allowed = {r.lower() for r in _enabled_rules(rules_csv)}
    for rule_id in _NOISY_RULES:
        if rule_id not in allowed:
            # The rule id may not exist in this pymarkdownlnt version; skip it.
            with contextlib.suppress(Exception):
                api.disable_rule_by_identifier(rule_id)

    try:
        result = api.scan_string(document)
    except PyMarkdownApiException as exc:
        logger.warning("markdown lint: pymarkdownlnt scan raised: %s", exc)
        return []
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("markdown lint: pymarkdownlnt scan failed: %s", exc)
        return []

    issues: list[LintIssue] = []
    for failure in result.scan_failures:
        rule_id = str(getattr(failure, "rule_id", "") or "").lower()
        if allowed and rule_id and rule_id not in allowed:
            continue
        line_number = int(getattr(failure, "line_number", 0) or 0)
        description = str(getattr(failure, "rule_description", "") or "") or str(
            getattr(failure, "detail", "") or ""
        )
        issues.append(LintIssue(rule_id=rule_id, line_number=line_number, description=description))
    return issues


def format_lint_issues(issues: list[LintIssue]) -> str:
    """Render a list of :class:`LintIssue` as a compact bulleted summary.

    Used both for the repair-prompt injection and the advisory HTML comment.
    """
    if not issues:
        return ""
    lines: list[str] = []
    for issue in issues:
        lines.append(f"- {issue.rule_id.upper()} (line {issue.line_number}): {issue.description}")
    return "\n".join(lines)


def lint_summary_comment(issues: list[LintIssue]) -> str:
    """Build a non-rendering HTML comment summarizing unresolved lint issues."""
    if not issues:
        return ""
    counts: dict[str, int] = {}
    for issue in issues:
        counts[issue.rule_id.upper()] = counts.get(issue.rule_id.upper(), 0) + 1
    parts = [f"{rule} x{count}" for rule, count in sorted(counts.items())]
    return f"<!-- lint warnings: {', '.join(parts)} -->"
