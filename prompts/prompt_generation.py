"""Belief initialization, incremental update, and text-generation prompts."""

from __future__ import annotations

import json


BELIEF_SYSTEM_PROMPT = """Maintain one task-conditioned Belief State.
Use Entity for concrete objects or actors, State for one Entity's own condition,
and Relation for a fact connecting two Entity/State nodes. Entity attributes may
contain only stable task-relevant scalar lookup values. Observed State/Relation
has probability 1 and no reason; inferred content has probability below 1 and a
reason. Keep only goal-relevant facts and never invent evidence. Return only the
requested JSON object."""


BELIEF_TEXT_SYSTEM_PROMPT = """Turn the supplied structured Belief into one concise,
coherent current-world description. Preserve the Goal, unresolved Gaps, observed
facts, and qualified inferred facts. Connect facts through Relations, do not add
new information, and return only {"belief_text": "..."}."""


INITIAL_SCHEMA = {
    "goal_specification": "concrete success condition",
    "entities": [
        {
            "entity_id": "e1",
            "entity_type": "type",
            "name": "name",
            "attributes": {},
        }
    ],
    "states": [
        {
            "state_id": "s1",
            "entity_id": "e1",
            "description": "one condition of e1",
            "probability": 1.0,
            "reason": None,
            "source_type": "observed",
        }
    ],
    "relations": [
        {
            "relation_id": "r1",
            "source_id": "e1",
            "target_id": "e2",
            "description": "directed connection",
            "probability": 1.0,
            "reason": None,
            "source_type": "observed",
        }
    ],
    "epistemic_gaps": [{"target": "unknown proposition", "reason": "relevance"}],
    "achievement_gaps": [
        {"target": "unfinished world condition", "reason": "discrepancy"}
    ],
    "frontier": {
        "gap_type": "epistemic or achievement",
        "target": "exact target from the selected gap",
    },
}


UPDATE_SCHEMA = {
    "remove_state_ids": [],
    "remove_relation_ids": [],
    "upsert_entities": [],
    "upsert_states": [],
    "upsert_relations": [],
    "epistemic_gaps": [{"target": "current unknown", "reason": "relevance"}],
    "achievement_gaps": [
        {"target": "current unfinished condition", "reason": "discrepancy"}
    ],
    "frontier": {
        "gap_type": "epistemic or achievement",
        "target": "keep the current target while its gap remains",
    },
}


EXECUTION_EXAMPLE = """Execution example: when new evidence directly establishes
a task-relevant state change, replace contradicted facts, remove only the Gaps
resolved by that evidence, and preserve every other unresolved Gap."""


DIAGNOSTIC_EXAMPLE = """Diagnostic example: a new negative test may lower an
inferred diagnosis probability while adding the observed test State. This is a
valid belief revision; preserve competing hypotheses not addressed by the test."""


def _task_mode_example(config: dict) -> str:
    """Task mode example."""

    mode = config.get("context", {}).get("task_mode", "execution")
    return DIAGNOSTIC_EXAMPLE if mode == "diagnostic" else EXECUTION_EXAMPLE


def build_initial_belief_prompt(
    state: dict,
    config: dict,
) -> list[dict[str, str]]:
    """Build initial belief prompt."""

    user = f"""Task:
{state["task"]}

Initial observation:
{state["observation"]}

Create the initial task-conditioned Belief. The acting agent is an Entity only
when it can act on and change the environment. Select one existing Gap as the
Frontier; use null only when no Gap exists.

{_task_mode_example(config)}

Required JSON schema:
{json.dumps(INITIAL_SCHEMA, ensure_ascii=False)}"""
    return [
        {"role": "system", "content": BELIEF_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_belief_update_prompt(
    belief_context: dict,
    transition: dict,
    state: dict,
    config: dict,
    consistency_issues: list[dict] | None = None,
    step: int | None = None,
    **_: object,
) -> list[dict[str, str]]:
    """Build belief update prompt."""

    issue_section = ""
    if consistency_issues:
        issue_section = (
            "\nThe previous candidate failed consistency validation. Correct "
            "only these issues using the supplied observation:\n"
            + json.dumps(consistency_issues, ensure_ascii=False)
        )
    user = f"""Current structured Belief:
{json.dumps(belief_context, ensure_ascii=False)}

Realized transition:
{json.dumps(transition, ensure_ascii=False)}
{issue_section}

Produce one incremental candidate update. Omitted World records remain unchanged;
remove a State or Relation only when the observation invalidates it. Return the
complete current Gap lists. Keep the existing Frontier while its Gap remains,
otherwise select an exact target from the new Gap lists.

{_task_mode_example(config)}

Required JSON schema:
{json.dumps(UPDATE_SCHEMA, ensure_ascii=False)}"""
    return [
        {"role": "system", "content": BELIEF_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_belief_text_prompt(
    semantic_view: dict,
    goal: dict,
    epistemic_gaps: list[dict],
    achievement_gaps: list[dict],
    config: dict,
    frontier_gap: dict | None = None,
    **_: object,
) -> list[dict[str, str]]:
    """Build belief text prompt."""

    user = json.dumps(
        {
            "goal": goal,
            "world": semantic_view,
            "epistemic_gaps": epistemic_gaps,
            "achievement_gaps": achievement_gaps,
            "active_gap": frontier_gap,
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": BELIEF_TEXT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_belief_repair_prompt(
    messages: list[dict[str, str]],
    raw_response: str,
    schema_error: str | None = None,
) -> list[dict[str, str]]:
    """Build belief repair prompt."""

    failure = (
        f"The previous JSON did not match the requested schema: {schema_error}. "
        if schema_error
        else "The previous response was not valid JSON. "
    )

    return messages + [
        {
            "role": "user",
            "content": (
                failure + "Return the same answer as one JSON object with exactly the "
                "requested keys; every array item must be a non-null object."
            ),
        }
    ]
