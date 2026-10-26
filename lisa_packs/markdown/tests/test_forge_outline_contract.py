"""Outline-contract gate for the forge: self-abridgment defense.

The 2026-10-16 incident: a forge on glm-5.3-flash:cloud declared four
planned sections in its document plan, stopped cleanly
(``truncated=False``, ``finish_reason=stop``), and quietly omitted the last
two sections. The continuation machinery correctly did nothing — the
provider signal was truthful — and the persisted document was short. The
outline-contract gate closes that gap: when the model declares a
``<!-- forge-plan: ... -->`` section plan and the finished document is
missing planned sections, bounded expansion passes write only the missing
sections. Documents without a plan marker skip the gate entirely, so the
legacy path is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from lisa_packs.markdown.operations._continuation import (
    missing_planned_sections,
    parse_forge_plan,
)
from lisa_packs.markdown.operations.forge_document.forge_document import (
    Parameters,
    execute,
)
from lisa_packs.markdown.tests.test_forge_document import (
    _fake_resolve_model,
    _FakeCompletion,
    _make_loader,
    _make_tool_executor,
)

# ── fixtures ────────────────────────────────────────────────────────────


@dataclass
class _FakeModel:
    provider: str = "openai"
    model_id: str = "gpt-chat-latest"
    display_name: str = "GPT Chat"


_DOC_NO_PLAN = "# Runbook\n\n## Symptoms\n- VM is unresponsive.\n\n## Steps\n1. Restart the VM.\n"


def _doc_with_plan(plan: str, body: str) -> str:
    """A fenced document whose first line declares ``plan``."""
    return f"```markdown\n<!-- forge-plan: {plan} -->\n{body}\n```"


_FULL_BODY = (
    "# Launch plan\n"
    "\n"
    "## Positioning\n"
    "Lead with what can be shown.\n"
    "\n"
    "## Evidence\n"
    "The thread archive is the artifact.\n"
    "\n"
    "## Launch sequence\n"
    "Publish, demo, onboard.\n"
    "\n"
    "## Competitive FAQ\n"
    "Answer each alternative with a differentiator.\n"
)

_PARTIAL_BODY = (
    "# Launch plan\n"
    "\n"
    "## Positioning\n"
    "Lead with what can be shown.\n"
    "\n"
    "## Evidence\n"
    "The thread archive is the artifact.\n"
)

_LAUNCH_SEQ_SECTION = (
    "## Launch sequence\n"
    "\n"
    "1. Publish the packages.\n"
    "2. Record the demo.\n"
    "3. Onboard contributors.\n"
)

_FAQ_SECTION = (
    "## Competitive FAQ\n"
    "\n"
    "- LangGraph: inspectable loop vs opaque graph.\n"
    "- CrewAI: exact-action approvals.\n"
)

_PLAN = "## Positioning / ## Evidence / ## Launch sequence / ## Competitive FAQ"

# Outline-contract attempt and completion-call counts the expansion paths
# are expected to make.
_SINGLE_ATTEMPT = 1
_TWO_ATTEMPTS = 2
_TWO_COMPLETION_CALLS = 2
_THREE_COMPLETION_CALLS = 3


def _document_from(content: str) -> _FakeCompletion:
    """A clean-stop completion carrying ``content`` inside a fence."""
    return _FakeCompletion(content=content, finish_reason="stop", truncated=False)


def _expansion_complete(sections: str) -> _FakeCompletion:
    """An expansion pass returning ``sections`` (already markdown)."""
    return _FakeCompletion(
        content=f"```markdown\n{sections}\n```",
        finish_reason="stop",
        truncated=False,
    )


def _make_ctx(
    complete_results: list[_FakeCompletion],
    *,
    params: dict[str, Any] | None = None,
    pack_config: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build a wired context plus the captures dict from the fake tools.

    ``pack_config`` seeds the session-manifest ``configuration.markdown``
    block the settings resolver reads (e.g. ``forge_outline_contract``).
    """
    from kernel.models import OperationExecutionContext
    from kernel.seams.host_adapter import _CallableHost

    tool_fn, captures = _make_tool_executor()
    calls: list[int] = []

    async def complete_fn(**kwargs: Any) -> _FakeCompletion:
        calls.append(int(kwargs.get("max_tokens") or 0))
        idx = min(len(calls) - 1, len(complete_results) - 1)
        return complete_results[idx]

    host = _CallableHost(
        complete_fn=complete_fn,
        resolve_model_fn=_fake_resolve_model(),
        read_authored_text_fn=_make_loader(),
        execute_tool_fn=tool_fn,
    )
    ctx = OperationExecutionContext(
        workspace_id="test",
        user_id="test-user",
        user_slug="test-user",
        surface_type="personal_dm",
        surface_id="test",
        memory_scope="personal",
        mode_id="explainer",
        query="forge a launch document",
        host=host,
        params=params or {},
        pack_config=pack_config,
    )
    calls.clear()
    captures["complete_calls"] = calls
    return ctx, captures


# ── parse_forge_plan ───────────────────────────────────────────────────


def test_parse_forge_plan_strips_marker_and_returns_sections() -> None:
    body, planned = parse_forge_plan(f"<!-- forge-plan: {_PLAN} -->\n{_PARTIAL_BODY}")
    assert planned == ("Positioning", "Evidence", "Launch sequence", "Competitive FAQ")
    assert body.strip() == _PARTIAL_BODY.strip()


def test_parse_forge_plan_no_marker_returns_empty_plan() -> None:
    body, planned = parse_forge_plan(_DOC_NO_PLAN)
    assert planned == ()
    assert body == _DOC_NO_PLAN


def test_parse_forge_plan_ignores_mid_document_marker() -> None:
    """A marker after body content is scaffolding drift — kept, unparsed."""
    doc = f"{_PARTIAL_BODY}\n<!-- forge-plan: ## Ghost -->\n"
    body, planned = parse_forge_plan(doc)
    assert planned == ()
    assert "forge-plan" in body


# ── missing_planned_sections ───────────────────────────────────────────


def test_missing_sections_fuzzy_matches_numbered_and_cased_headings() -> None:
    planned = ("Launch sequence", "Competitive FAQ", "Evidence")
    document = (
        "# Launch plan\n\n"
        "## 4. Launch Sequence\nbody\n\n"
        "## 5. Competitive FAQ — why not just X?\nbody\n\n"
        "## Evidence\nbody\n"
    )
    assert missing_planned_sections(planned, document) == ()


def test_missing_sections_reports_absent_titles() -> None:
    planned = ("Positioning", "Launch sequence", "Competitive FAQ")
    assert missing_planned_sections(planned, _PARTIAL_BODY) == (
        "Launch sequence",
        "Competitive FAQ",
    )


def test_missing_sections_empty_plan_is_empty() -> None:
    assert missing_planned_sections((), _PARTIAL_BODY) == ()


def test_missing_sections_top_level_heading_does_not_satisfy() -> None:
    """A lone ``#`` title does not satisfy a planned ``##`` section."""
    document = "# Launch sequence\nbody\n"
    assert missing_planned_sections(("Launch sequence",), document) == ("Launch sequence",)


# ── execute() gate behavior ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_plan_marker_takes_legacy_path_with_no_extra_calls() -> None:
    """No plan declaration → gate skipped, exactly one provider call."""
    ctx, captures = _make_ctx([_document_from(f"```markdown\n{_FULL_BODY}\n```")])
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    assert result.status == "success"
    assert captures["complete_calls"] == [50_000]
    assert result.extra["outline_contract"]["stop_reason"] == "skipped"
    assert result.extra["outline_contract"]["planned"] == []
    assert result.extra["outline_contract"]["missing_after"] == []
    persisted = captures["writes"][0]["content"]
    assert "forge-plan" not in persisted


@pytest.mark.asyncio
async def test_satisfied_plan_persists_with_no_extra_calls() -> None:
    """Plan declared and fully honored → gate passes, one provider call."""
    ctx, captures = _make_ctx([_document_from(_doc_with_plan(_PLAN, _FULL_BODY))])
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    assert result.status == "success"
    assert captures["complete_calls"] == [50_000]
    assert result.extra["outline_contract"]["stop_reason"] == "satisfied"
    persisted = captures["writes"][0]["content"]
    assert "forge-plan" not in persisted
    assert "## Competitive FAQ" in persisted


@pytest.mark.asyncio
async def test_missing_sections_expand_and_persist_complete_document() -> None:
    """Two missing sections → two expand passes → complete document."""
    ctx, captures = _make_ctx(
        [
            _document_from(_doc_with_plan(_PLAN, _PARTIAL_BODY)),
            _expansion_complete(_LAUNCH_SEQ_SECTION),
            _expansion_complete(_FAQ_SECTION),
        ]
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    assert result.status == "success"
    assert len(captures["writes"]) == 1
    persisted = captures["writes"][0]["content"]
    assert "forge-plan" not in persisted
    assert "## Launch sequence" in persisted
    assert "## Competitive FAQ" in persisted
    assert persisted.index("## Evidence") < persisted.index("## Launch sequence")
    contract = result.extra["outline_contract"]
    assert contract["missing_before"] == ["Launch sequence", "Competitive FAQ"]
    assert contract["missing_after"] == []
    assert contract["stop_reason"] == "expanded"
    assert contract["attempts"] == _TWO_ATTEMPTS
    # First forge pass + two expansion passes.
    assert len(captures["complete_calls"]) == _THREE_COMPLETION_CALLS


@pytest.mark.asyncio
async def test_expansion_still_missing_persists_degraded_with_warning() -> None:
    """Expansion passes yield nothing useful → degraded, warning summary."""
    ctx, captures = _make_ctx(
        [
            _document_from(_doc_with_plan(_PLAN, _PARTIAL_BODY)),
            _expansion_complete(""),
            _expansion_complete(""),
        ]
    )
    result = await execute(ctx)
    # The summary leads with the incompleteness, not with "Document forged":
    # a caller (or a model skimming the result) that reads only the first
    # clause must not mistake a truncated artifact for a complete one.
    assert result.output_text.startswith("INCOMPLETE:")
    assert "Document forged" not in result.output_text
    assert result.status == "degraded"
    assert result.output_json["complete"] is False
    assert "Launch sequence" in result.output_text
    assert "Competitive FAQ" in result.output_text
    contract = result.extra["outline_contract"]
    assert contract["missing_after"] == ["Launch sequence", "Competitive FAQ"]
    assert contract["stop_reason"] == "still_missing"
    # Both budgeted attempts ran; each returned an empty pass and the gate
    # gave up after exhausting the budget.
    assert contract["attempts"] == _TWO_ATTEMPTS
    persisted = captures["writes"][0]["content"]
    assert "missing sections: Launch sequence, Competitive FAQ" in persisted


@pytest.mark.asyncio
async def test_whole_tail_fallback_merges_tail_once() -> None:
    """A heading-less expansion tail merges once, never once per title.

    Two missing sections and a tail without any planned heading used to
    append the same prose twice (once per remaining title) under the
    per-title loop. The whole-tail fallback is single-shot.
    """
    loose = "1. Publish the packages.\n2. Record the demo.\n3. Answer each alternative.\n"
    ctx, captures = _make_ctx(
        [
            _document_from(_doc_with_plan(_PLAN, _PARTIAL_BODY)),
            _expansion_complete(loose),
        ]
    )
    result = await execute(ctx)
    persisted = captures["writes"][0]["content"]
    assert persisted.count("1. Publish the packages.") == 1
    contract = result.extra["outline_contract"]
    # The loose tail satisfied neither planned heading in name, so the
    # sections stay missing — but the prose merged exactly once: pass 1
    # appended it, pass 2 detected the duplicate and aborted.
    assert contract["missing_after"] == ["Launch sequence", "Competitive FAQ"]
    assert contract["attempts"] == _TWO_ATTEMPTS


@pytest.mark.asyncio
async def test_single_pass_writes_both_missing_sections() -> None:
    """One expansion pass covering both missing sections → one extra call."""
    both = f"{_LAUNCH_SEQ_SECTION}\n\n{_FAQ_SECTION}\n"
    ctx, captures = _make_ctx(
        [
            _document_from(_doc_with_plan(_PLAN, _PARTIAL_BODY)),
            _expansion_complete(both),
        ]
    )
    result = await execute(ctx)
    assert result.status == "success"
    persisted = captures["writes"][0]["content"]
    assert "## Launch sequence" in persisted
    assert "## Competitive FAQ" in persisted
    contract = result.extra["outline_contract"]
    assert contract["missing_after"] == []
    assert contract["attempts"] == _SINGLE_ATTEMPT
    assert len(captures["complete_calls"]) == _TWO_COMPLETION_CALLS


@pytest.mark.asyncio
async def test_gate_disabled_skips_expansion_calls() -> None:
    """``forge_outline_contract: false`` → plan parsed but no expand passes."""
    ctx, captures = _make_ctx(
        [_document_from(_doc_with_plan(_PLAN, _PARTIAL_BODY))],
        pack_config={"forge_outline_contract": False},
    )
    result = await execute(ctx)
    assert "Document forged" in result.output_text
    assert result.status == "success"
    assert captures["complete_calls"] == [50_000]
    assert result.extra["outline_contract"]["stop_reason"] == "disabled"
    persisted = captures["writes"][0]["content"]
    assert "## Launch sequence" not in persisted


@pytest.mark.asyncio
async def test_plan_marker_never_reaches_persisted_document() -> None:
    """The scaffolding comment is stripped even when the gate is disabled."""
    ctx, captures = _make_ctx(
        [_document_from(_doc_with_plan(_PLAN, _PARTIAL_BODY))],
        pack_config={"forge_outline_contract": False},
    )
    await execute(ctx)
    persisted = captures["writes"][0]["content"]
    assert "forge-plan" not in persisted
    assert "## Positioning" in persisted


def test_parameters_model_importable() -> None:
    """The Parameters model stays importable for the SDK catalog builder."""
    assert Parameters is not None
