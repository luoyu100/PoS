"""Task-policy prompts and action protocols for Raw/ReAct and PoS."""

from __future__ import annotations

from copy import deepcopy
import json
import re


SYSTEM_PROMPT = """You are an agent that solves interactive tasks with ReAct.
At each step, reason about the current task and observation, then choose exactly one action.
Use the thought to state what the latest observation established and what you still need to resolve.
The action will be sent directly to the benchmark environment, so copy its syntax exactly.
Return only one JSON object with this format:
{"thought": "brief reasoning", "action": "one action"}
Do not add Markdown fences or any text outside the JSON object.
"""


LOCA_BENCH_SYSTEM_PROMPT = """You are an agent that solves interactive tasks with ReAct.
Choose exactly one available tool whose arguments match its schema.
Use tool observations as evidence and call claim_done only after the requested outcome is complete.
Return only:
{"thought": "brief reasoning", "action": {"name": "tool name", "arguments": {}}}
Do not add Markdown or other text.
"""


REPAIR_PROMPT = (
    "Return the same decision as one complete JSON object in the required schema."
)


RCA100_FINAL_TURN_DIRECTIVE = """This is the final interaction turn: no further investigation is possible.
Call submit_root_cause now with your best evidence-supported conclusion."""


CLINDIAG_FINAL_TURN_DIRECTIVE = """This is the final interaction turn: no further information can be obtained.
Call submit_diagnosis now with your best evidence-supported diagnosis."""


def final_turn_directive(config: dict) -> str:
    """Final turn directive."""

    if config["benchmark"] == "RCA100":
        return RCA100_FINAL_TURN_DIRECTIVE
    if config["benchmark"] == "ClinDiag":
        return CLINDIAG_FINAL_TURN_DIRECTIVE
    return ""


def append_final_turn_directive(
    messages: list[dict[str, str]],
    config: dict,
) -> list[dict[str, str]]:
    """Append final turn directive."""

    directive = final_turn_directive(config)
    if not directive:
        return messages
    return [
        *messages[:-1],
        {
            **messages[-1],
            "content": f"{messages[-1]['content']}\n{directive}\n",
        },
    ]


RCA100_SYSTEM_PROMPT_HEADER = """You are an SRE agent that performs root cause analysis with ReAct.
At each turn, reason about the incident and the latest observations, then issue tool calls.
Use the thought to state what the latest observations established and what you still need to rule out.
"""


RCA100_SINGLE_ACTION_FORMAT = """Return only one JSON object with this format:
{"thought": "brief reasoning", "action": {"name": "tool name", "arguments": {}}}
Every response must contain both keys "thought" and "action"; a response without an executable action is invalid.
"""


RCA100_MULTI_ACTION_FORMAT = """Return only one JSON object with this format:
{{"thought": "brief reasoning", "actions": [{{"name": "tool name", "arguments": {{}}}}, ...]}}
"actions" is an ordered list of 1 to {k} tool calls; they are executed sequentially and each call returns its own observation before your next turn.
Batch up to {k} independent queries into one turn for efficiency; keep dependent queries (whose arguments need an earlier result) for the next turn.
Always call submit_root_cause alone, as the only action of its turn.
Every response must contain both keys "thought" and "actions"; a response without an executable action is invalid.
"""


RCA100_SYSTEM_PROMPT_RULES = """The action name must exactly match an available tool, and arguments must follow that tool's parameter schema.
The Available actions list is exhaustive and authoritative; a tool not in the list does not exist.
The observability data is not included in the input; inspect it only through the tools, starting from the alert entity and its topology neighborhood, then cross-check metrics, logs, traces, and events inside the alert window.
Calling a tool does not prove anything by itself; use the returned observation as the evidence.
If a tool result is truncated, narrow the query (entity, time range, keyword, or limit) instead of repeating it.
The alert entity is the symptom entry point and is frequently not the root cause; follow the propagation upstream to the true origin.
Call submit_root_cause exactly once, with the best evidence-supported root-cause entity and fault type.
Do not add Markdown fences or any text outside the JSON object.
"""


RCA100_REASONING_EXAMPLE = """Reasoning example: correlated symptoms are not
direct evidence; use available tools to test the suspected causal mechanism
before submitting."""


def rca100_actions_per_turn(config: dict) -> int:
    """Rca100 actions per turn."""

    return int(config.get("max_actions_per_turn", 1))


def rca100_system_prompt(config: dict) -> str:
    """Rca100 system prompt."""

    k = rca100_actions_per_turn(config)
    action_format = (
        RCA100_MULTI_ACTION_FORMAT.format(k=k) if k > 1 else RCA100_SINGLE_ACTION_FORMAT
    )
    return (
        f"{RCA100_SYSTEM_PROMPT_HEADER}{action_format}"
        f"{RCA100_SYSTEM_PROMPT_RULES}\n{rca100_task_info()}"
    )


CLINDIAG_SYSTEM_PROMPT = """You are a doctor agent that performs the clinical diagnostic procedure with ReAct.
At each turn, reason about the patient and the latest answers, then choose exactly one available tool.
Use the thought to state what the latest answers established and what you still need to rule out.
Return only one JSON object with this format:
{"thought": "brief reasoning", "action": {"name": "tool name", "arguments": {}}}
Every response must contain both keys "thought" and "action"; a response without an executable action is invalid.
The Available actions list is exhaustive and authoritative; a tool not in the list does not exist.
The patient record is not included in the input; obtain it only through the tools, one specific question per turn.
Call submit_diagnosis exactly once, with the most specific evidence-supported diagnosis.
Do not add Markdown fences or any text outside the JSON object.
"""


CLINDIAG_EXAMPLE = """Fictional format-only example, unrelated to benchmark cases:
Task: {"initial_presentation": "An adult patient reports a new symptom."}
Observation: The patient is in front of you; only the initial presentation is known so far.
Response: {"thought": "I should establish the symptom timeline.", "action": {"name": "ask_history", "arguments": {"question": "When did the symptom begin?"}}}
Final-turn format, only after sufficient evidence; replace every placeholder with findings from the current case:
Response: {"thought": "The collected evidence supports a diagnosis.", "action": {"name": "submit_diagnosis", "arguments": {"final_diagnosis": "<specific evidence-supported diagnosis>", "differential_diagnosis": "<ranked alternatives>", "diagnostic_reasoning": "<supporting observations and reasons for excluding alternatives>"}}}
"""


def clindiag_task_info() -> str:
    """Clindiag task info."""

    from benchmarks.ClinDiag.info import (
        CLINICAL_WORKFLOW_INFO,
        QUESTIONING_DISCIPLINE_INFO,
        ROLE_CONTRACT_INFO,
        SUBMIT_CONTRACT_INFO,
    )

    return (
        f"{ROLE_CONTRACT_INFO}\n\n"
        f"{CLINICAL_WORKFLOW_INFO}\n\n"
        f"{QUESTIONING_DISCIPLINE_INFO}\n\n"
        f"{SUBMIT_CONTRACT_INFO}"
    )


def rca100_task_info() -> str:
    """Rca100 task info."""

    from benchmarks.RCA100.info import (
        DIAGNOSTIC_METHOD_INFO,
        ENTITY_STANDARD_INFO,
        FAULT_TYPES_INFO,
        SUBMIT_CONTRACT_INFO,
    )

    return (
        f"{FAULT_TYPES_INFO}\n\n"
        f"{ENTITY_STANDARD_INFO}\n\n"
        f"{DIAGNOSTIC_METHOD_INFO}\n\n"
        f"{SUBMIT_CONTRACT_INFO}"
    )


def benchmark_protocol_info(config: dict) -> str:
    """Benchmark protocol info."""

    if config["benchmark"] == "RCA100":
        return f"{RCA100_SYSTEM_PROMPT_RULES}\n{rca100_task_info()}"
    if config["benchmark"] == "ClinDiag":
        return clindiag_task_info()
    return ""


def benchmark_role_prefix(config: dict) -> str:
    """Benchmark role prefix."""

    if config["benchmark"] == "RCA100":
        return "You are an SRE agent that performs root cause analysis"
    if config["benchmark"] == "ClinDiag":
        return "You are a doctor agent that performs the clinical diagnostic procedure"
    return ""


ALFWORLD_EXAMPLE = """Example:
Task: put a clean bowl in a cabinet
Observation: You are in a kitchen. A countertop 1 and cabinet 1 are visible.
Available actions: ["go to countertop 1", "go to cabinet 1", "look"]
Response: {"thought": "I should inspect likely locations for the bowl first.", "action": "go to countertop 1"}
"""


LOCA_BENCH_EXAMPLE = """Example:
Task: find the upcoming exam for course CS101 and notify the enrolled students
Observation: The task environment is ready. Use the available tools to inspect and update it.
Available actions: [{"type": "function", "function": {"name": "canvas_list_assignments", "description": "List assignments for a course.", "parameters": {"type": "object", "properties": {"course_id": {"type": "string"}}, "required": ["course_id"]}}}]
Response: {"thought": "I should inspect the course assignments before sending any notification.", "action": {"name": "canvas_list_assignments", "arguments": {"course_id": "CS101"}}}
"""


FRONTIER_ACTION_SELECTION_CONTRACT = """Choose an action that directly resolves or
narrows the Active Gap; use a prerequisite action only when a required argument
for such an action is still unknown."""


_LOCA_TOOL_CATEGORY_ORDER = (
    "metadata_discovery",
    "evidence_retrieval",
    "deterministic_compute",
    "side_effect",
    "finish",
    "other",
)


def _content_tokens(text: object) -> set[str]:
    """Content tokens."""

    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "gap_type",
        "is",
        "of",
        "or",
        "reason",
        "target",
        "the",
        "to",
    }
    return {
        token
        for token in re.findall(r"\w+", str(text).casefold())
        if token not in stopwords
    }


def _tool_domain(name: str, description: str) -> str:
    """Recover the MCP server label embedded by LOCA in each description."""

    match = re.match(r"\s*\[([^\]]+)\]", description)
    if match is not None:
        return match.group(1)
    return name.split("_", maxsplit=1)[0] if name else "unknown"


def _tool_category(name: str, domain: str) -> str:
    """Group tools by stable interface role, never by benchmark task words."""

    normalized = name.lower()
    tokens = set(re.split(r"[^a-z0-9]+", normalized))
    if "claim_done" in normalized or domain == "claim_done":
        return "finish"
    if "python_execute" in normalized or domain in {"python_execute", "terminal"}:
        return "deterministic_compute"
    if tokens & {
        "create",
        "update",
        "write",
        "send",
        "delete",
        "remove",
        "add",
        "upload",
        "load",
        "export",
        "cancel",
    }:
        return "side_effect"
    if (
        tokens & {"list", "search", "metadata", "schema", "info", "describe"}
        or "get_dataset_info" in normalized
    ):
        return "metadata_discovery"
    if tokens & {"read", "query", "get", "fetch", "download", "lookup"}:
        return "evidence_retrieval"
    return "other"


def build_loca_tool_catalog(actions: list[dict]) -> dict:
    """Group complete tool definitions by role without shortening any fields."""

    domains: dict[str, int] = {}
    grouped: dict[str, list[dict]] = {
        category: [] for category in _LOCA_TOOL_CATEGORY_ORDER
    }
    for action in actions:
        function = action.get("function", {})
        name = str(function.get("name", ""))
        if not name:
            continue
        description = str(function.get("description", ""))
        domain = _tool_domain(name, description)
        category = _tool_category(name, domain)
        domains[domain] = domains.get(domain, 0) + 1

        grouped[category].append(deepcopy(function))

    return {
        "domains": [
            {"name": name, "tool_count": count}
            for name, count in sorted(domains.items())
        ],
        "tools_by_role": {
            category: sorted(grouped[category], key=lambda item: item["name"])
            for category in _LOCA_TOOL_CATEGORY_ORDER
            if grouped[category]
        },
    }


def _frontier_relevant_world(world: object, frontier: object) -> dict:
    """Frontier relevant world."""

    empty = {"entities": {}, "states": {}, "relations": {}}
    if not isinstance(world, dict) or not isinstance(frontier, dict):
        return empty
    entities = world.get("entities", {})
    states = world.get("states", {})
    relations = world.get("relations", {})
    if not all(isinstance(items, dict) for items in (entities, states, relations)):
        return empty

    focus = _content_tokens(json.dumps(frontier, ensure_ascii=False))
    entity_ids = {
        str(entity_id)
        for entity_id, entity in entities.items()
        if isinstance(entity, dict)
        and focus
        & _content_tokens(
            " ".join(
                [
                    str(entity_id),
                    str(entity.get("name", "")),
                    str(entity.get("entity_type", "")),
                ]
            )
        )
    }
    state_ids = {
        str(state_id)
        for state_id, state in states.items()
        if isinstance(state, dict)
        and (
            str(state.get("entity_id", "")) in entity_ids
            or focus & _content_tokens(f"{state_id} {state.get('description', '')}")
        )
    }
    references = entity_ids | state_ids
    selected_relations = {
        str(relation_id): relation
        for relation_id, relation in relations.items()
        if isinstance(relation, dict)
        and (
            str(relation.get("source_id", "")) in references
            or str(relation.get("target_id", "")) in references
            or focus
            & _content_tokens(f"{relation_id} {relation.get('description', '')}")
        )
    }
    for relation in selected_relations.values():
        references.update(
            {
                str(relation.get("source_id", "")),
                str(relation.get("target_id", "")),
            }
        )
    entity_ids |= references & set(map(str, entities))
    state_ids |= references & set(map(str, states))
    state_ids |= {
        str(state_id)
        for state_id, state in states.items()
        if isinstance(state, dict) and str(state.get("entity_id", "")) in entity_ids
    }
    return {
        "entities": {
            key: value for key, value in entities.items() if str(key) in entity_ids
        },
        "states": {
            key: value for key, value in states.items() if str(key) in state_ids
        },
        "relations": selected_relations,
    }


def react_system_prompt(config: dict) -> str:
    """React system prompt."""

    if config["benchmark"] == "LOCA-Bench":
        return LOCA_BENCH_SYSTEM_PROMPT
    if config["benchmark"] == "RCA100":
        return rca100_system_prompt(config)
    if config["benchmark"] == "ClinDiag":
        return f"{CLINDIAG_SYSTEM_PROMPT}\n{clindiag_task_info()}"
    return SYSTEM_PROMPT


def react_example(config: dict) -> str:
    """React example."""

    if config["benchmark"] == "ALFWorld":
        return ALFWORLD_EXAMPLE
    if config["benchmark"] == "LOCA-Bench":
        return LOCA_BENCH_EXAMPLE
    if config["benchmark"] == "RCA100":
        return RCA100_REASONING_EXAMPLE
    if config["benchmark"] == "ClinDiag":
        return CLINDIAG_EXAMPLE
    return ""


def build_prompt(
    state: dict,
    trajectory: list[dict],
    config: dict,
) -> list[dict[str, str]]:
    """Build prompt."""

    example = react_example(config)

    history = json.dumps(trajectory, ensure_ascii=False, indent=2)
    actions = state["actions"] if config["include_actions"] else []
    if config["benchmark"] == "LOCA-Bench":
        actions = build_loca_tool_catalog(actions)
    user_prompt = f"""{example}
Task:
{state["task"]}

Current observation:
{state["observation"]}

Available actions:
{json.dumps(actions, ensure_ascii=False)}

Previous steps:
{history}

Choose the next action and return the required JSON object.
"""

    return [
        {"role": "system", "content": react_system_prompt(config)},
        {"role": "user", "content": user_prompt},
    ]


def build_belief_prompt(
    state: dict,
    belief_context: dict,
    config: dict,
) -> list[dict[str, str]]:
    """Build belief prompt."""

    example = "" if config["benchmark"] == "RCA100" else react_example(config)
    example_section = f"{example}\n" if example else ""
    actions = state["actions"] if config["include_actions"] else []
    if config["benchmark"] == "LOCA-Bench":
        action_view = build_loca_tool_catalog(actions)
    else:
        action_view = actions

    primary_frontier = belief_context.get("frontier_gap")
    structured = {
        "goal": belief_context.get("goal"),
        "world": _frontier_relevant_world(
            belief_context.get("world", {}),
            primary_frontier,
        ),
    }
    recovery = belief_context.get("recovery")
    runtime = state.get("benchmark_info", {}).get("workspace")
    runtime_section = f"\nRuntime workspace: {runtime}\n" if runtime else ""
    recovery_section = (
        "\nRecovery constraint C_t:\n" + json.dumps(recovery, ensure_ascii=False)
        if recovery is not None
        else ""
    )
    user_prompt = f"""{example_section}Current observation:
{state["observation"]}

Belief Text:
{belief_context["belief_text"]}

Goal and Active-Gap structured subgraph:
{json.dumps(structured, ensure_ascii=False)}

Active Gap:
{json.dumps(primary_frontier, ensure_ascii=False)}

{FRONTIER_ACTION_SELECTION_CONTRACT}
{recovery_section}
{runtime_section}
Available action schemas:
{json.dumps(action_view, ensure_ascii=False)}

Choose one action and return the required JSON object."""
    return [
        {"role": "system", "content": react_system_prompt(config)},
        {"role": "user", "content": user_prompt},
    ]


def build_repair_prompt(
    messages: list[dict[str, str]],
    raw_response: str,
) -> list[dict[str, str]]:
    """Build repair prompt."""

    return [
        *messages,
        {"role": "assistant", "content": raw_response},
        {"role": "user", "content": REPAIR_PROMPT},
    ]
