"""Stable gap identities and windowed Belief Trapping detection."""

from __future__ import annotations

import re
import json
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


GapType = Literal["epistemic", "achievement"]
_GAP_TYPES: tuple[GapType, ...] = ("epistemic", "achievement")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "gap",
    "gap_type",
    "is",
    "of",
    "or",
    "reason",
    "target",
    "the",
    "to",
}


def normalize_gap_text(text: object) -> str:
    """Normalize gap case and whitespace without creating new identities."""

    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def _tokens(text: object) -> set[str]:
    """Extract content tokens for conservative semantic alignment."""

    return {
        token
        for token in re.findall(
            r"\w+",
            normalize_gap_text(text),
            flags=re.UNICODE,
        )
        if token not in _STOPWORDS
    }


def _read(record: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dataclass record or a dictionary."""

    return (
        record.get(key, default)
        if isinstance(record, Mapping)
        else getattr(record, key, default)
    )


def _gap_target(gap: Any) -> str:
    """Read the target from either supported gap representation."""

    return str(_read(gap, "target", ""))


def _gap_reason(gap: Any) -> str:
    """Read the explanation from either supported gap representation."""

    return str(_read(gap, "reason", ""))


def gaps_semantically_equivalent(
    left: Any,
    right: Any,
    *,
    gap_type: GapType | None = None,
    entity_names: Any = None,
    minimum_score: float = 0.8,
) -> bool:
    """Check whether adjacent gaps express the same unresolved proposition."""

    left_target = normalize_gap_text(_gap_target(left))
    right_target = normalize_gap_text(_gap_target(right))
    if left_target == right_target:
        return True
    left_tokens = _tokens(left_target)
    right_tokens = _tokens(right_target)
    if min(len(left_tokens), len(right_tokens)) < 2:
        return False
    shared = left_tokens & right_tokens
    overlap = len(shared) / min(len(left_tokens), len(right_tokens))
    jaccard = len(shared) / len(left_tokens | right_tokens)
    return (0.7 * overlap + 0.3 * jaccard) >= minimum_score


@dataclass(frozen=True, slots=True)
class StableGap:
    """Episode-local stable semantic gap identity."""

    identity: str
    gap_type: GapType
    target: str
    reason: str


class StableGapIdentityTracker:
    """Align gap identities between adjacent committed belief updates."""

    def __init__(self, minimum_score: float = 0.8):
        self.minimum_score = minimum_score
        self._counters = {"epistemic": 0, "achievement": 0}
        self._active: dict[str, tuple[StableGap, ...]] = {
            "epistemic": (),
            "achievement": (),
        }

    def reset(self) -> None:
        """Clear episode-local gap identities."""

        self._counters = {"epistemic": 0, "achievement": 0}
        self._active = {"epistemic": (), "achievement": ()}

    def reconcile(
        self,
        gap_type: GapType,
        gaps: Sequence[Any],
        *,
        entity_names: Any = None,
    ) -> tuple[StableGap, ...]:
        """Reuse uniquely matched gap identities and allocate identities for new gaps."""

        previous = list(self._active[gap_type])
        used: set[int] = set()
        tracked: list[StableGap] = []
        for gap in gaps:
            matches = [
                index
                for index, old in enumerate(previous)
                if index not in used
                and gaps_semantically_equivalent(
                    old,
                    gap,
                    gap_type=gap_type,
                    minimum_score=self.minimum_score,
                )
            ]
            if len(matches) == 1:
                index = matches[0]
                identity = previous[index].identity
                used.add(index)
            else:
                self._counters[gap_type] += 1
                prefix = "e" if gap_type == "epistemic" else "a"
                identity = f"{prefix}:{self._counters[gap_type]}"
            tracked.append(
                StableGap(
                    identity,
                    gap_type,
                    _gap_target(gap),
                    _gap_reason(gap),
                )
            )
        self._active[gap_type] = tuple(tracked)
        return self._active[gap_type]

    def active(self, gap_type: GapType) -> tuple[StableGap, ...]:
        """Return stable gaps currently present in the requested dimension."""

        return self._active[gap_type]

    def resolve(
        self,
        gap_type: GapType,
        target: str,
        *,
        entity_names: Any = None,
    ) -> StableGap | None:
        """Resolve a surface target to one current stable gap."""

        matches = [
            gap
            for gap in self.active(gap_type)
            if gaps_semantically_equivalent(
                gap,
                {"target": target},
                gap_type=gap_type,
                minimum_score=self.minimum_score,
            )
        ]
        return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True, slots=True)
class TrappingDetection:
    """Windowed health values and their trapping classification."""

    window_ready: bool
    window_start_step: int | None = None
    window_end_step: int | None = None
    persistence_e: float = 0.0
    persistence_a: float = 0.0
    stagnation: float = 0.0
    recurrence: float = 0.0
    recurrence_period: int | None = None
    health_score: float = 1.0
    health_threshold: float = 0.25
    gap_dimension: GapType | None = None
    trapping_pattern: Literal["static", "cycle", "drift"] | None = None
    persistent_epistemic_gaps: tuple[str, ...] = ()
    persistent_achievement_gaps: tuple[str, ...] = ()

    def to_belief_health_kwargs(self) -> dict[str, Any]:
        """Convert detection results to BeliefHealth constructor fields."""

        return {
            "window_ready": self.window_ready,
            "persistence_e": self.persistence_e,
            "persistence_a": self.persistence_a,
            "stagnation": self.stagnation,
            "recurrence": self.recurrence,
            "recurrence_period": self.recurrence_period,
            "health_score": self.health_score,
            "health_threshold": self.health_threshold,
            "gap_dimension": self.gap_dimension,
            "trapping_pattern": self.trapping_pattern,
            "persistent_epistemic_gaps": self.persistent_epistemic_gaps,
            "persistent_achievement_gaps": self.persistent_achievement_gaps,
        }


def detect_belief_trapping(
    transitions: Sequence[Any],
    snapshots: Sequence[Mapping[str, Any]],
    *,
    active_gap: Mapping[str, Any] | None,
    window_size: int,
    health_threshold: float,
    cycle_threshold: float,
    recurrence_epsilon: float,
    max_cycle_length: int,
) -> TrappingDetection:
    """Compute trapping over the latest K verified transitions across frontiers."""

    if not 1 <= max_cycle_length < window_size:
        raise ValueError("Require 1 <= max_cycle_length < window_size")
    if not all(
        0 <= value <= 1
        for value in (health_threshold, cycle_threshold, recurrence_epsilon)
    ):
        raise ValueError("Health and recurrence thresholds must be in [0, 1]")
    verified = [
        transition
        for transition in transitions
        if _read(transition, "progress") in (0, 1)
    ]
    if len(verified) < window_size:
        return TrappingDetection(window_ready=False, health_threshold=health_threshold)
    window = verified[-window_size:]
    steps = [int(_read(item, "step")) for item in window]
    snapshots_by_step = {int(item["step"]): item for item in snapshots}
    try:
        beliefs = [snapshots_by_step[step] for step in steps]
    except KeyError:
        return TrappingDetection(window_ready=False, health_threshold=health_threshold)

    persistent_e, persistence_e = _persistence(beliefs, "epistemic")
    persistent_a, persistence_a = _persistence(beliefs, "achievement")
    stagnation = (
        1.0 - sum(int(_read(item, "progress")) for item in window) / window_size
    )

    projections = [
        _active_gap_projection(item.get("world", {}), active_gap) for item in beliefs
    ]
    recurrence, period = _recurrence(
        projections,
        recurrence_epsilon,
        max_cycle_length,
    )
    health_score = 1.0 - max(persistence_e, persistence_a) * max(stagnation, recurrence)

    dimension: GapType | None = None
    pattern: Literal["static", "cycle", "drift"] | None = None
    if health_score <= health_threshold:
        candidates = [
            (kind, persistence)
            for kind, persistence in (
                ("epistemic", persistence_e),
                ("achievement", persistence_a),
            )
            if beliefs[-1].get(f"{kind}_gap_ids")
        ]
        if candidates:
            dimension = max(candidates, key=lambda item: item[1])[0]
        full_distance = _world_distance(
            beliefs[-1].get("world", {}),
            beliefs[-2].get("world", {}),
        )
        active_distance = _projection_distance(projections[-1], projections[-2])
        if full_distance <= recurrence_epsilon:
            pattern = "static"
        elif recurrence >= cycle_threshold and period is not None and period > 1:
            pattern = "cycle"
        elif active_distance <= recurrence_epsilon:
            pattern = "drift"

    return TrappingDetection(
        window_ready=True,
        window_start_step=steps[0],
        window_end_step=steps[-1],
        persistence_e=persistence_e,
        persistence_a=persistence_a,
        stagnation=stagnation,
        recurrence=recurrence,
        recurrence_period=period,
        health_score=health_score,
        health_threshold=health_threshold,
        gap_dimension=dimension,
        trapping_pattern=pattern,
        persistent_epistemic_gaps=tuple(sorted(persistent_e)),
        persistent_achievement_gaps=tuple(sorted(persistent_a)),
    )


def _persistence(
    snapshots: Sequence[Mapping[str, Any]],
    gap_type: GapType,
) -> tuple[set[str], float]:
    """Compute the fraction of initial gaps unresolved throughout the window."""

    key = f"{gap_type}_gap_ids"
    sets = [set(snapshot.get(key, ())) for snapshot in snapshots]
    initial = sets[0]
    if not initial:
        return set(), 0.0
    persistent = set.intersection(*sets)
    return persistent, len(persistent) / len(initial)


def _active_gap_projection(
    world: Mapping[str, Any],
    active_gap: Mapping[str, Any] | None,
) -> dict[str, set[str]]:
    """Project the world to entities, states, and relations relevant to the active gap."""

    entities = world.get("entities", {})
    states = world.get("states", {})
    relations = world.get("relations", {})
    if active_gap is None:
        return _world_sets(world)
    focus = _tokens(active_gap.get("target", "")) | _tokens(
        active_gap.get("reason", "")
    )
    selected_entities = {
        entity_id
        for entity_id, entity in entities.items()
        if _tokens(entity.get("name", "")) & focus
        or _tokens(entity.get("entity_type", "")) & focus
    }
    selected_states = {
        state_id
        for state_id, state in states.items()
        if state.get("entity_id") in selected_entities
        or _tokens(state.get("description", "")) & focus
    }
    selected_entities.update(
        states[state_id].get("entity_id")
        for state_id in selected_states
        if states[state_id].get("entity_id")
    )
    selected_relations = {
        relation_id
        for relation_id, relation in relations.items()
        if relation.get("source_id") in selected_entities | selected_states
        or relation.get("target_id") in selected_entities | selected_states
        or _tokens(relation.get("description", "")) & focus
    }
    for relation_id in selected_relations:
        relation = relations[relation_id]
        for node_id in (relation.get("source_id"), relation.get("target_id")):
            if node_id in entities:
                selected_entities.add(node_id)
            if node_id in states:
                selected_states.add(node_id)
    return {
        "entities": {_semantic_record(entities[item]) for item in selected_entities},
        "states": {_semantic_record(states[item]) for item in selected_states},
        "relations": {_semantic_record(relations[item]) for item in selected_relations},
    }


def _world_sets(world: Mapping[str, Any]) -> dict[str, set[str]]:
    """Convert the world into three comparable semantic record sets."""

    return {
        kind: {_semantic_record(record) for record in world.get(kind, {}).values()}
        for kind in ("entities", "states", "relations")
    }


def _semantic_record(record: Mapping[str, Any]) -> str:
    """Compare aligned records including confidence, independently of field order."""

    return normalize_gap_text(
        json.dumps(dict(record), sort_keys=True, ensure_ascii=False)
    )


def _recurrence(
    projections: Sequence[Mapping[str, set[str]]],
    epsilon: float,
    max_cycle_length: int,
) -> tuple[float, int | None]:
    """Measure recurrence from the active-gap world projection alone."""

    best_score = 0.0
    best_lag: int | None = None
    for lag in range(1, min(max_cycle_length, len(projections) - 1) + 1):
        matches = [
            _projection_distance(
                projections[index],
                projections[index - lag],
            )
            <= epsilon
            for index in range(lag, len(projections))
        ]
        score = sum(matches) / len(matches)
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_score, best_lag


def _projection_distance(
    left: Mapping[str, set[str]],
    right: Mapping[str, set[str]],
) -> float:
    """Average entity, state, and relation Jaccard distances."""

    return (
        sum(
            _jaccard_distance(left.get(kind, set()), right.get(kind, set()))
            for kind in ("entities", "states", "relations")
        )
        / 3.0
    )


def _world_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Compute semantic distance between complete world projections."""

    return _projection_distance(_world_sets(left), _world_sets(right))


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    """Return Jaccard distance, with zero distance for two empty sets."""

    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)
