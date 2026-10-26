"""Reasoning-burnout rescue for the forge first pass.

The 2026-10-14 forge incident: ``target_length: 1000`` derived a
1300-token first-pass budget; the reasoning model (glm-5.3:cloud) spent
its whole budget on chain-of-thought before emitting any document text
(``finish_reason=length, content=""``), the operation returned the bare
"Empty Response" error every retry, and the failure was misdiagnosed as
environmental. The rescue retries the same call once at the operation's
flat ceiling and surfaces the budget diagnostics on the failure path;
both behaviors mirror the turn-loop rescues in ``kernel.turn_runner``
via the shared ``kernel.model_budgets`` predicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from kernel.models import OperationExecutionContext

from lisa_packs.markdown.operations.forge_document.forge_document import (
    _burnout_retry_budget,
    execute,
)
from lisa_packs.markdown.tests.test_forge_document import (
    _VALID_DOCUMENT,
    _fake_resolve_model,
    _FakeCompletion,
    _make_loader,
    _make_tool_executor,
)

# The flat ceiling the operation retries at, and the budgets under test.
_FLAT_CEILING = 50_000
_MODEL_CEILING = 8192
_FIRST_PASS_BUDGET = 1300
_TARGET_LENGTH = 1000


@dataclass
class _BurnoutModel:
    """A reasoning model with an optional per-model completion ceiling.

    ``max_completion_tokens`` 0 (the manifest's "no declaration") leaves
    the rescue bounded only by the operation's flat ceiling.
    """

    provider: str = "ollama"
    model_id: str = "glm-5.3:cloud"
    display_name: str = "GLM"
    reasoning_model: bool = True
    thinking_model: bool = False
    max_completion_tokens: int = 0
    context_window: int = 262144


def _burnout_completion(*, max_tokens: int, finish_reason: str = "length") -> _FakeCompletion:
    """One budget-exhausted empty completion at ``max_tokens``.

    ``finish_reason='length'`` with empty content is the provider's own
    exhaustion proof. The clean-stop variant needs the burn-share spend,
    which the fake supplies via ``completion_tokens`` so
    ``_is_reasoning_burnout_completion`` classifies it.
    """
    return _FakeCompletion(
        content="",
        finish_reason=finish_reason,
        truncated=False if finish_reason == "stop" else None,
        completion_tokens=max_tokens,
    )


def _forged_completion() -> _FakeCompletion:
    return _FakeCompletion(content=f"```markdown\n{_VALID_DOCUMENT}\n```")


def _make_ctx_with(
    complete_fn: Any,
    *,
    params: dict[str, Any] | None = None,
    model_def: Any = None,
) -> tuple[OperationExecutionContext, dict[str, Any]]:
    """Build a wired context and its captures dict (writes/outline tooling)."""
    from kernel.seams.host_adapter import _CallableHost

    tool_fn, captures = _make_tool_executor()
    resolve = _fake_resolve_model() if model_def is None else (lambda _alias: model_def)
    host = _CallableHost(
        complete_fn=complete_fn,
        resolve_model_fn=resolve,
        read_authored_text_fn=_make_loader(),
        execute_tool_fn=tool_fn,
    )
    return (
        OperationExecutionContext(
            workspace_id="test",
            user_id="test-user",
            user_slug="test-user",
            surface_type="personal_dm",
            surface_id="test",
            memory_scope="personal",
            mode_id="explainer",
            query="forge a design document",
            host=host,
            params=params or {},
            pack_config=None,
        ),
        captures,
    )


def _recording_complete(results: list[_FakeCompletion], *, max_tokens_seen: list[int]) -> Any:
    """A ``complete_fn`` returning ``results`` in order, recording budgets."""

    async def _complete(**kwargs: Any) -> _FakeCompletion:
        max_tokens_seen.append(int(kwargs.get("max_tokens") or 0))
        idx = min(len(results) - 1, len(max_tokens_seen) - 1)
        return results[idx]

    return _complete


# ── _burnout_retry_budget ──────────────────────────────────────────────


def test_burnout_retry_budget_jumps_to_flat_ceiling() -> None:
    """A derived first-pass budget rescues to the operation's flat ceiling."""

    assert (
        _burnout_retry_budget(
            first_pass_max_tokens=_FIRST_PASS_BUDGET,
            model_def=_BurnoutModel(),
            flat_ceiling=_FLAT_CEILING,
        )
        == _FLAT_CEILING
    )


def test_burnout_retry_budget_respects_model_ceiling() -> None:
    """A tighter declared model ceiling caps the rescue."""

    assert (
        _burnout_retry_budget(
            first_pass_max_tokens=_FIRST_PASS_BUDGET,
            model_def=_BurnoutModel(max_completion_tokens=_MODEL_CEILING),
            flat_ceiling=_FLAT_CEILING,
        )
        == _MODEL_CEILING
    )


def test_burnout_retry_budget_none_when_no_higher_budget() -> None:
    """No rescue when the flat ceiling cannot exceed the exhausted budget."""

    assert (
        _burnout_retry_budget(
            first_pass_max_tokens=_FLAT_CEILING,
            model_def=_BurnoutModel(),
            flat_ceiling=_FLAT_CEILING,
        )
        is None
    )


def test_burnout_retry_budget_none_for_plain_model_budget() -> None:
    """A model without ``max_completion_tokens`` leaves the flat ceiling alone."""

    plain = _BurnoutModel(max_completion_tokens=0)
    assert (
        _burnout_retry_budget(
            first_pass_max_tokens=_FIRST_PASS_BUDGET, model_def=plain, flat_ceiling=_FLAT_CEILING
        )
        == _FLAT_CEILING
    )


# ── execute() rescue behavior ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_burnout_first_pass_retries_at_flat_ceiling() -> None:
    """``finish_reason=length`` + empty content retries once at the ceiling."""
    max_tokens_seen: list[int] = []
    complete_fn = _recording_complete(
        [_burnout_completion(max_tokens=_FIRST_PASS_BUDGET), _forged_completion()],
        max_tokens_seen=max_tokens_seen,
    )
    ctx, captures = _make_ctx_with(complete_fn, params={"target_length": _TARGET_LENGTH})
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    assert max_tokens_seen[0] == _FIRST_PASS_BUDGET
    assert max_tokens_seen[1] == _FLAT_CEILING
    assert len(captures["writes"]) == 1


@pytest.mark.asyncio
async def test_clean_stop_burnout_retries_too() -> None:
    """The clean-stop burnout shape (stop + whole budget on reasoning) retries."""
    burnout = _FakeCompletion(
        content="",
        finish_reason="stop",
        truncated=False,
        reasoning_content="extended deliberation about the document structure",
        completion_tokens=_FIRST_PASS_BUDGET,
    )
    max_tokens_seen: list[int] = []
    complete_fn = _recording_complete(
        [burnout, _forged_completion()], max_tokens_seen=max_tokens_seen
    )
    ctx, _captures = _make_ctx_with(complete_fn, params={"target_length": _TARGET_LENGTH})
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    assert max_tokens_seen == [_FIRST_PASS_BUDGET, _FLAT_CEILING]


@pytest.mark.asyncio
async def test_cheap_empty_response_retries_once_then_fails_loudly() -> None:
    """A cheap clean stop is retried once and then reported as a failure.

    Regression for the 2026-10-26 incident: an empty body with
    ``finish_reason='stop'`` and sparse spend. The shared 90% burnout share
    deliberately declines this shape, so the operation must own the gate — a
    document forge has no valid empty outcome.

    The retry escalates to the flat ceiling per the spec's §4.4, matching
    every other rescue path in this operation.
    """
    max_tokens_seen: list[int] = []
    complete_fn = _recording_complete(
        [
            _FakeCompletion(content="", finish_reason="stop", completion_tokens=12),
            _FakeCompletion(content="", finish_reason="stop", completion_tokens=11),
        ],
        max_tokens_seen=max_tokens_seen,
    )
    ctx, _captures = _make_ctx_with(complete_fn, params={"target_length": _TARGET_LENGTH})
    result = await execute(ctx)

    assert "Empty Response" in result.output_text
    assert result.status == "error", "an empty forge must never report success"
    assert result.error_detail and "finish_reason='stop'" in result.error_detail
    # Two calls: the first pass and one escalated retry.
    assert max_tokens_seen == [_FIRST_PASS_BUDGET, _FLAT_CEILING]


@pytest.mark.asyncio
async def test_cheap_empty_response_succeeds_on_retry() -> None:
    """The spec's case 1: empty clean stop, low spend, then success."""
    max_tokens_seen: list[int] = []
    complete_fn = _recording_complete(
        [
            _FakeCompletion(content="", finish_reason="stop", completion_tokens=12),
            _forged_completion(),
        ],
        max_tokens_seen=max_tokens_seen,
    )
    ctx, _captures = _make_ctx_with(complete_fn, params={"target_length": _TARGET_LENGTH})
    result = await execute(ctx)

    assert "Document forged and stored" in result.output_text
    assert max_tokens_seen == [_FIRST_PASS_BUDGET, _FLAT_CEILING]


@pytest.mark.asyncio
async def test_empty_response_includes_budget_diagnostics() -> None:
    """The failure path names the finish reason and the budget that failed.

    After a rescue the diagnostics report the *retried* budget — the spend
    that finally failed — which is what an operator needs to tell a
    budget-shape failure (raise the ceiling) from an input failure.
    """
    max_tokens_seen: list[int] = []
    complete_fn = _recording_complete(
        [_burnout_completion(max_tokens=_FIRST_PASS_BUDGET)],
        max_tokens_seen=max_tokens_seen,
    )
    ctx, _captures = _make_ctx_with(complete_fn, params={"target_length": _TARGET_LENGTH})
    result = await execute(ctx)
    assert "Empty Response" in result.output_text
    assert "finish_reason=length" in result.output_text
    # The rescue raised to the flat ceiling first; the failure reports that
    # budget and the burnout classification.
    assert "max_tokens=50000" in result.output_text
    assert "likely_reasoning_budget_exhaustion" in result.output_text
    assert max_tokens_seen == [_FIRST_PASS_BUDGET, _FLAT_CEILING]


@pytest.mark.asyncio
async def test_rescue_failure_reports_retry_ceiling() -> None:
    """When the rescue also returns empty, the diagnostics show the retried budget."""
    complete_fn = _recording_complete(
        [
            _burnout_completion(max_tokens=_FIRST_PASS_BUDGET),
            _burnout_completion(max_tokens=_FLAT_CEILING),
        ],
        max_tokens_seen=[],
    )
    ctx, _captures = _make_ctx_with(complete_fn, params={"target_length": _TARGET_LENGTH})
    result = await execute(ctx)
    assert "Empty Response" in result.output_text
    assert "max_tokens=50000" in result.output_text


@pytest.mark.asyncio
async def test_rescue_disabled_by_config() -> None:
    """``forge_burnout_rescue: false`` keeps the legacy single-call behavior."""
    max_tokens_seen: list[int] = []
    complete_fn = _recording_complete(
        [_burnout_completion(max_tokens=_FIRST_PASS_BUDGET)],
        max_tokens_seen=max_tokens_seen,
    )
    ctx, _captures = _make_ctx_with(
        complete_fn,
        params={"target_length": _TARGET_LENGTH},
        model_def=_BurnoutModel(),
    )
    ctx = ctx.model_copy(update={"pack_config": {"forge_burnout_rescue": False}})
    result = await execute(ctx)
    assert max_tokens_seen == [_FIRST_PASS_BUDGET]
    assert "Empty Response" in result.output_text


@pytest.mark.asyncio
async def test_rescue_budget_capped_by_model_ceiling() -> None:
    """A model-declared ceiling below the flat ceiling binds the retry."""
    max_tokens_seen: list[int] = []
    complete_fn = _recording_complete(
        [_burnout_completion(max_tokens=_FIRST_PASS_BUDGET), _forged_completion()],
        max_tokens_seen=max_tokens_seen,
    )
    ctx, _captures = _make_ctx_with(
        complete_fn,
        params={"target_length": _TARGET_LENGTH},
        model_def=_BurnoutModel(max_completion_tokens=_MODEL_CEILING),
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    assert max_tokens_seen == [_FIRST_PASS_BUDGET, _MODEL_CEILING]


@pytest.mark.asyncio
async def test_no_rescue_when_flat_ceiling_not_higher() -> None:
    """A first pass already at the flat ceiling cannot be rescued."""
    complete_fn = _recording_complete(
        [_burnout_completion(max_tokens=_FLAT_CEILING)],
        max_tokens_seen=[],
    )
    ctx, _captures = _make_ctx_with(complete_fn, params={})
    result = await execute(ctx)
    assert "Empty Response" in result.output_text
    assert "max_tokens=50000" in result.output_text


@pytest.mark.asyncio
async def test_healthy_first_pass_unchanged() -> None:
    """A normal forge result byte-behaves as before the rescue existed."""
    max_tokens_seen: list[int] = []
    complete_fn = _recording_complete([_forged_completion()], max_tokens_seen=max_tokens_seen)
    ctx, captures = _make_ctx_with(complete_fn, params={"target_length": _TARGET_LENGTH})
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    assert max_tokens_seen == [_FIRST_PASS_BUDGET]
    assert len(captures["writes"]) == 1
