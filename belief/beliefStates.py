"""PoS runtime representation B_t = (W_t, G, Delta_E, Delta_A).

Structured belief is the source of truth; belief text is a derived decision view."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from belief.worldStates import WorldState


@dataclass(slots=True)
class Goal:
    """Fixed goal G: the user request and its interpreted specification."""

    user_input: str
    specification: str


@dataclass(slots=True)
class EpistemicGap:
    """Task-relevant information that remains unknown or unverified."""

    target: str
    reason: str


@dataclass(slots=True)
class AchievementGap:
    """An unresolved difference between the current and desired world."""

    target: str
    reason: str


@dataclass(slots=True)
class GapFrontier:
    """Pointer to the single gap currently prioritized for investigation or action."""

    gap_type: Literal["epistemic", "achievement"]
    target: str


@dataclass(slots=True)
class HealthTransition:
    """Action-observation-belief transition with its verification and progress result."""

    step: int
    frontier_episode: int
    frontier: GapFrontier | None
    action: Any
    observation: str
    done: bool = False
    success: bool = False
    task_score: float = 0.0
    progress: int | None = None
    scoring_method: str | None = None
    score: float | None = None
    affected_record_ids: tuple[str, ...] = ()
    reason: str = ""


@dataclass(slots=True)
class BeliefHealth:
    """Health H_t: persistence, stagnation, recurrence, and trapping classification."""

    window_ready: bool = False
    persistence_e: float = 0.0
    persistence_a: float = 0.0
    stagnation: float = 0.0
    recurrence: float = 0.0
    recurrence_period: int | None = None
    health_score: float = 1.0
    health_threshold: float = 0.25
    gap_dimension: Literal["epistemic", "achievement"] | None = None
    trapping_pattern: Literal["static", "cycle", "drift"] | None = None
    persistent_epistemic_gaps: tuple[str, ...] = ()
    persistent_achievement_gaps: tuple[str, ...] = ()

    @property
    def trapped(self) -> bool:
        """Return whether the validated window crosses the health threshold."""

        return self.window_ready and self.health_score <= self.health_threshold


@dataclass(slots=True)
class RecoveryDirective:
    """Temporary recovery constraints C_t supplied to the next task-policy decision."""

    recovery_frontier: GapFrontier
    trigger_frontier: GapFrontier | None
    gap_dimension: Literal["epistemic", "achievement"]
    trapping_pattern: Literal["static", "cycle", "drift"] | None
    started_at_step: int
    applies_from_step: int
    recovery_action_steps: list[int] = field(default_factory=list)
    low_progress_transitions: list[dict[str, Any]] = field(default_factory=list)
    instruction: str = ""


@dataclass(slots=True)
class BeliefState:
    """Structured belief, active gap, health, and derived natural-language context."""

    world: WorldState
    goal: Goal
    epistemic_gaps: list[EpistemicGap] = field(default_factory=list)
    achievement_gaps: list[AchievementGap] = field(default_factory=list)
    frontier: GapFrontier | None = None
    health: BeliefHealth = field(default_factory=BeliefHealth)
    belief_text: str = ""
    step: int = 0

    def advance_step(self) -> None:
        """Advance the belief interaction index after an environment interaction."""

        self.step += 1
