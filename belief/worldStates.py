"""Entity-State-Relation representation of the current task world.

Entities carry stable lookup attributes. States describe one entity; relations
connect entities or states. Observed facts have unit confidence and no reason."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ENTITY_ATTRIBUTES_MAX_CHARS = 4096


def entity_attributes_problem(attributes: object) -> str | None:
    """Return an invariant violation for noncompact or nonscalar entity attributes."""

    if not isinstance(attributes, dict):
        return "attributes must be a JSON object"
    for key, value in attributes.items():
        if not isinstance(key, str) or not key.strip():
            return "attribute keys must be non-empty strings"
        if value is not None and not isinstance(value, (str, int, float, bool)):
            return (
                f"attribute {key!r} must be a scalar or null, not "
                f"{type(value).__name__}"
            )
    serialized = json.dumps(
        attributes,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(serialized) > ENTITY_ATTRIBUTES_MAX_CHARS:
        return (
            "attributes exceed the compact representation budget of "
            f"{ENTITY_ATTRIBUTES_MAX_CHARS} characters"
        )
    return None


@dataclass(slots=True)
class Entity:
    """Task-world object, process, or event with stable lookup attributes."""

    entity_id: str  # Start with char "e", e.g. e1, e2
    entity_type: str
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate scalar attributes; null denotes removal of a previously stored key."""

        problem = entity_attributes_problem(self.attributes)
        if problem is not None:
            raise TypeError(problem)


@dataclass(slots=True)
class State:
    """Natural-language belief about the intrinsic state of one entity."""

    state_id: str  # Start with char "s", e.g. s1, s2
    entity_id: str
    description: str
    probability: float = 1.0
    reason: str | None = None
    source_type: Literal["observed", "inferred"] = "observed"


@dataclass(slots=True)
class Relation:
    """Directed structural, evidential, or causal belief between two records."""

    relation_id: str  # Start with char "r", e.g. r1, r2
    source_id: str
    target_id: str
    description: str
    probability: float = 0.5
    reason: str | None = None
    source_type: Literal["observed", "inferred"] = "observed"


########################################################
#              World States Representation             #
########################################################


@dataclass(slots=True)
class WorldState:
    """Current world W_t represented by entities, states, and relations."""

    entities: dict[str, Entity] = field(default_factory=dict)
    states: dict[str, State] = field(default_factory=dict)
    relations: dict[str, Relation] = field(default_factory=dict)

    def upsert_entity(self, entity: Entity) -> None:
        """Merge attributes for an existing ID; omitted keys retain their prior values."""

        previous = self.entities.get(entity.entity_id)
        merged_attributes = dict(previous.attributes) if previous is not None else {}
        for key, value in entity.attributes.items():
            if value is None:
                merged_attributes.pop(key, None)
            else:
                merged_attributes[key] = value
        entity.attributes = merged_attributes
        self.entities[entity.entity_id] = entity

    def upsert_state(self, state: State) -> None:
        """Store an entity state; isolated states need not participate in a relation."""

        if state.entity_id not in self.entities:
            raise ValueError(f"State owner entity does not exist: {state.entity_id}")

        if state.source_type == "observed":
            state.probability = 1.0
            state.reason = None
        self.states[state.state_id] = state

    def upsert_relation(self, relation: Relation) -> None:
        """Store a relation after checking that both endpoint records exist."""

        if not self._reference_exists(relation.source_id):
            raise ValueError(f"Relation source does not exist: {relation.source_id}")
        if not self._reference_exists(relation.target_id):
            raise ValueError(f"Relation target does not exist: {relation.target_id}")

        if relation.source_type == "observed":
            relation.probability = 1.0
            relation.reason = None
        self.relations[relation.relation_id] = relation

    def remove_state(self, state_id: str) -> None:
        """Remove a state and all relations that reference it; deletion is idempotent."""

        self.states.pop(state_id, None)
        self.relations = {
            relation_id: relation
            for relation_id, relation in self.relations.items()
            if relation.source_id != state_id and relation.target_id != state_id
        }

    def remove_relation(self, relation_id: str) -> None:
        """Remove a relation if present."""

        self.relations.pop(relation_id, None)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible representation used by prompts and event traces."""

        entities = {}
        for entity_id, entity in self.entities.items():
            payload = asdict(entity)

            if not payload["attributes"]:
                payload.pop("attributes")
            entities[entity_id] = payload
        states = {state_id: asdict(state) for state_id, state in self.states.items()}
        relations = {
            relation_id: asdict(relation)
            for relation_id, relation in self.relations.items()
        }
        return {
            "entities": entities,
            "states": states,
            "relations": relations,
        }

    def _reference_exists(self, reference_id: str) -> bool:
        """Return whether an ID refers to a current entity or state."""

        return reference_id in self.entities or reference_id in self.states
