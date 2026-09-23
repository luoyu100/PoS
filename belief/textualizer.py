"""Build connected semantic fact groups for belief text generation."""

from __future__ import annotations

from typing import Any

from belief.worldStates import Entity, Relation, State, WorldState


def _entity_fields(entity: Entity) -> dict[str, Any]:
    """Expose entity identity and compact lookup attributes."""

    fields: dict[str, Any] = {
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "name": entity.name,
    }
    if entity.attributes:
        fields["attributes"] = dict(entity.attributes)
    return fields


def _belief_fields(belief: State | Relation) -> dict[str, Any]:
    """Extract shared description, confidence, and evidence-type fields."""

    return {
        "description": belief.description,
        "probability": belief.probability,
        "reason": belief.reason,
        "source_type": belief.source_type,
    }


def _owner_entity_id(world: WorldState, reference_id: str) -> str:
    """Resolve an entity or state reference to its owning entity."""

    if reference_id in world.entities:
        return reference_id
    return world.states[reference_id].entity_id


def _reference_view(world: WorldState, reference_id: str) -> dict[str, Any]:
    """Expand a relation endpoint so the language model need not resolve record IDs."""

    if reference_id in world.entities:
        entity = world.entities[reference_id]
        return {
            "kind": "entity",
            **_entity_fields(entity),
        }

    state = world.states[reference_id]
    return {
        "kind": "state",
        "state_id": state.state_id,
        "entity_id": state.entity_id,
        **_belief_fields(state),
    }


def _connected_entity_groups(world: WorldState) -> list[list[str]]:
    """Group connected entities while retaining isolated entities."""

    adjacency = {entity_id: set() for entity_id in world.entities}
    for relation in world.relations.values():
        source_id = _owner_entity_id(world, relation.source_id)
        target_id = _owner_entity_id(world, relation.target_id)
        adjacency[source_id].add(target_id)
        adjacency[target_id].add(source_id)

    ordered_entities = sorted(
        world.entities,
        key=lambda entity_id: (
            world.entities[entity_id].entity_type.lower() != "agent",
            entity_id,
        ),
    )
    remaining = set(ordered_entities)
    groups = []

    for root_id in ordered_entities:
        if root_id not in remaining:
            continue

        group = []
        frontier = [root_id]
        remaining.remove(root_id)
        while frontier:
            entity_id = frontier.pop()
            group.append(entity_id)
            for neighbor_id in sorted(adjacency[entity_id]):
                if neighbor_id in remaining:
                    remaining.remove(neighbor_id)
                    frontier.append(neighbor_id)
        groups.append(sorted(group))

    return groups


def build_semantic_view(world: WorldState) -> dict[str, Any]:
    """Assemble entities, states, and relations into connected semantic fact groups."""

    states_by_entity = {entity_id: [] for entity_id in world.entities}
    for state_id in sorted(world.states):
        state = world.states[state_id]
        states_by_entity[state.entity_id].append(
            {
                "state_id": state.state_id,
                **_belief_fields(state),
            }
        )

    relation_entities = {
        relation_id: {
            _owner_entity_id(world, relation.source_id),
            _owner_entity_id(world, relation.target_id),
        }
        for relation_id, relation in world.relations.items()
    }
    groups = []

    for index, entity_ids in enumerate(_connected_entity_groups(world), start=1):
        entity_id_set = set(entity_ids)
        entities = []
        for entity_id in entity_ids:
            entity = world.entities[entity_id]
            entities.append(
                {
                    **_entity_fields(entity),
                    "states": states_by_entity[entity_id],
                }
            )

        relations = []
        for relation_id in sorted(world.relations):
            if not relation_entities[relation_id].issubset(entity_id_set):
                continue
            relation = world.relations[relation_id]
            relations.append(
                {
                    "relation_id": relation.relation_id,
                    "source": _reference_view(world, relation.source_id),
                    "target": _reference_view(world, relation.target_id),
                    **_belief_fields(relation),
                }
            )

        groups.append(
            {
                "group_id": index,
                "entities": entities,
                "relations": relations,
            }
        )

    return {"fact_groups": groups}
