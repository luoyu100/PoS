"""Compose pattern-specific and gap-specific constraints without selecting actions."""

from __future__ import annotations

import json
from typing import Any


def compose_recovery_instruction(
    *,
    pattern: str | None,
    gap_type: str,
    gap: dict,
    recent_transitions: list[dict[str, Any]],
    recurrence_period: int | None,
    cycle_transitions: list[dict[str, Any]] | None = None,
) -> str:
    """Compose C_pattern union C_gap without choosing or executing an action."""

    pattern_constraint = _pattern_constraint(
        pattern,
        recent_transitions,
        recurrence_period,
        cycle_transitions or [],
    )
    if gap_type == "epistemic":
        gap_constraint = (
            "Acquire new discriminative evidence that can resolve or narrow "
            f"the epistemic recovery gap: {gap['target']}"
        )
    else:
        gap_constraint = (
            "Cause a task-relevant world-state change that reduces this "
            f"achievement recovery gap: {gap['target']}"
        )
    return f"{pattern_constraint} {gap_constraint}"


def _pattern_constraint(
    pattern: str | None,
    recent_transitions: list[dict[str, Any]],
    recurrence_period: int | None,
    cycle_transitions: list[dict[str, Any]],
) -> str:
    """Translate static, cycle, or drift dynamics into a local recovery constraint."""

    if pattern == "static":
        action = recent_transitions[-1].get("action") if recent_transitions else None
        suffix = f" ({json.dumps(action, ensure_ascii=False)})" if action else ""
        return (
            "Do not repeat the latest ineffective action under the unchanged "
            f"belief condition{suffix}."
        )
    if pattern == "cycle":
        period = recurrence_period or 2
        return (
            f"Avoid the transition that re-enters the detected {period}-step "
            "cycle; choose a valid exit from the recurrent region. "
            f"Matched transitions: {json.dumps(cycle_transitions, ensure_ascii=False)}."
        )
    if pattern == "drift":
        return (
            "Reject off-gap actions and re-anchor the next decision to the Active Gap."
        )
    return "Reassess the current approach and choose an evidence-supported action toward the Active Gap."
