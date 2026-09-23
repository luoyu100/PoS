"""Rca100 env utilities."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Callable

from benchmarks.RCA100.info import FAULT_TYPE_IDS

SUBMIT_TOOL_NAME = "submit_root_cause"


SUBMIT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": SUBMIT_TOOL_NAME,
        "description": (
            "Submit the final root cause analysis and end the case. "
            "Call it exactly once, only after the evidence is sufficient. "
            "fault_type must be one name from the fault type catalog; "
            "root_cause_entity must be one service-level or node-level "
            "entity name copied verbatim from the observability data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "root_cause_entity": {
                    "type": "string",
                    "description": (
                        "Exactly one root-cause entity name: a microservice "
                        "name (apm.service) or a Kubernetes node name "
                        "(k8s.node), verbatim from the data."
                    ),
                },
                "fault_type": {
                    "type": "string",
                    "description": (
                        "Exactly one fault type name from the closed "
                        "catalog, e.g. nodeCpuHigh."
                    ),
                    "enum": FAULT_TYPE_IDS,
                },
                "analysis": {
                    "type": "string",
                    "description": (
                        "Brief evidence-backed reasoning for the conclusion."
                    ),
                },
            },
            "required": ["root_cause_entity", "fault_type"],
        },
    },
}


class RCA100Case:
    """RCA100Case."""

    def __init__(self, case_dir: Path):
        """Initialize configuration and episode-local runtime state."""

        self.case_dir = Path(case_dir)
        self.task = json.loads(
            (self.case_dir / "task.json").read_text(encoding="utf-8")
        )
        self._frames: dict[str, Any] = {}
        self._topology: dict | None = None

    def topology(self) -> dict:
        """Topology."""

        if self._topology is None:
            self._topology = json.loads(
                (self.case_dir / "topology.json").read_text(encoding="utf-8")
            )
        return self._topology

    def metrics(self):
        """Metrics."""

        return self._frame("metrics")

    def logs(self):
        """Logs."""

        return self._frame("logs")

    def traces(self):
        """Traces."""

        return self._frame("traces")

    def events(self):
        """Events."""

        return self._frame("events")

    def alerts(self):
        """Alerts."""

        return self._frame("alerts")

    def release(self) -> None:
        """Release."""

        self._frames.clear()
        self._topology = None

    def _frame(self, name: str):
        """Frame."""

        if name not in self._frames:
            import pandas as pd

            self._frames[name] = pd.read_parquet(self.case_dir / f"{name}.parquet")
        return self._frames[name]


class RCA100ToolRegistry:
    """RCA100Tool Registry."""

    def __init__(self, tools_module_path: Path | None):
        """Initialize configuration and episode-local runtime state."""

        self._functions: dict[str, Callable[..., Any]] = {}
        self._schemas: list[dict] = []
        submit_schema = SUBMIT_TOOL_SCHEMA

        if tools_module_path is not None and Path(tools_module_path).is_file():
            for entry in self._load_tools(Path(tools_module_path)):
                schema = entry["schema"]
                name = schema["function"]["name"]
                if name == SUBMIT_TOOL_NAME:
                    submit_schema = schema
                    continue
                if name in self._functions:
                    raise ValueError(f"Duplicate RCA100 tool name: {name}")
                self._functions[name] = entry["function"]
                self._schemas.append(schema)

        self._schemas.append(submit_schema)

    @staticmethod
    def _load_tools(module_path: Path) -> list[dict]:
        """Load tools."""

        spec = importlib.util.spec_from_file_location(
            "rca100_user_tools",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tools = getattr(module, "TOOLS", None)
        if not isinstance(tools, list):
            raise ValueError(f"{module_path} must define a module-level list TOOLS")
        for entry in tools:
            if (
                not isinstance(entry, dict)
                or "schema" not in entry
                or "function" not in entry
                or not callable(entry["function"])
                or entry["schema"].get("type") != "function"
                or "name" not in entry["schema"].get("function", {})
            ):
                raise ValueError(
                    "Each TOOLS entry must be "
                    '{"schema": <OpenAI function schema>, "function": callable}'
                )
        return tools

    def schemas(self) -> list[dict]:
        """Schemas."""

        return list(self._schemas)

    def has_tool(self, name: str) -> bool:
        """Has tool."""

        return name in self._functions

    def dispatch(self, name: str, arguments: dict, case: RCA100Case) -> str:
        """Dispatch."""

        problem = _answer_key_access_problem(arguments)
        if problem is not None:
            return f"Tool call rejected: {problem}"
        try:
            return str(self._functions[name](case, **arguments))
        except TypeError as error:
            return (
                f"Tool call error: arguments do not match the {name} "
                f"parameter schema ({error})."
            )
        except Exception as error:
            return f"Tool execution error in {name}: {error}"


def _answer_key_access_problem(arguments: dict) -> str | None:
    """Answer key access problem."""

    for key, value in arguments.items():
        if isinstance(value, str) and "answer_key" in value.lower():
            return (
                f"argument {key!r} references answer_key, which is "
                "not part of the agent-facing environment"
            )
    return None


def cap_observation(observation: str, max_chars: int) -> str:
    """Cap observation."""

    if len(observation) <= max_chars:
        return observation
    return (
        observation[:max_chars]
        + f"\n... [observation truncated at {max_chars} characters; "
        "total length was "
        f"{len(observation)}. Narrow the query instead of repeating it.]"
    )


def normalize_entity(name: object) -> str:
    """Normalize entity."""

    return re.sub(r"\s+", " ", str(name).strip()).lower()


def normalize_fault_type(value: object) -> str | None:
    """Normalize fault type."""

    text = re.sub(r"^[Ff]\d+-", "", str(value).strip()).lower()
    if not text:
        return None
    for fault_type in FAULT_TYPE_IDS:
        if text == fault_type.lower():
            return fault_type
    return None


def validate_submission(arguments: dict) -> list[str]:
    """Validate submission."""

    problems = []
    entity = arguments.get("root_cause_entity")
    if not isinstance(entity, str) or not entity.strip():
        problems.append("root_cause_entity must be one non-empty entity name string")
    if normalize_fault_type(arguments.get("fault_type", "")) is None:
        problems.append("fault_type must be exactly one ID from the fault type catalog")
    return problems


def score_submission(arguments: dict, answer_key_dir: Path, task_id: str) -> dict:
    """Score submission."""

    ground_truth = json.loads(
        (Path(answer_key_dir) / f"{task_id}.gt.json").read_text(encoding="utf-8")
    )
    expected_entities = ground_truth["root_cause_entities"]
    expected_types = [
        normalized
        for normalized in (
            normalize_fault_type(value)
            for value in ground_truth.get("root_cause_types", [])
        )
        if normalized is not None
    ]

    submitted_entity = str(arguments["root_cause_entity"])
    submitted_type = normalize_fault_type(arguments["fault_type"])
    entity_correct = normalize_entity(submitted_entity) in {
        normalize_entity(name) for name in expected_entities
    }
    type_correct = submitted_type in expected_types
    success = entity_correct and type_correct

    return {
        "submitted_entity": submitted_entity,
        "submitted_fault_type": submitted_type,
        "analysis": str(arguments.get("analysis", "")),
        "expected_entities": list(expected_entities),
        "expected_fault_types": expected_types,
        "entity_correct": entity_correct,
        "type_correct": type_correct,
        "success": success,
        "score": float(success),
    }
