"""Unified reset, step, result, and resource-management interface for four benchmarks."""

from __future__ import annotations

import json
import re
import os
import shutil
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any

from benchmarks.loca_artifacts import (
    append_workspace_output_manifest,
    capture_excel_artifact_contract,
    materialize_loca_observation,
    validated_workspace_output,
)
from benchmarks.loca_capabilities import attach_loca_capabilities

ROOT = Path(__file__).resolve().parents[1]
LOCA_BENCH_COMMIT = "8b6fac49d9edd92922593e703b74ea255357c3ec"


class Adapter:
    """Common benchmark facade with lazy benchmark-specific dependencies."""

    def __init__(self, config: dict):
        """Initialize configuration and episode-local runtime state."""

        if config["benchmark"] == "ALFWorld":
            self.benchmark = _ALFWorld(config)
        elif config["benchmark"] == "LOCA-Bench":
            self.benchmark = _LOCABench(config)
        elif config["benchmark"] == "RCA100":
            self.benchmark = _RCA100(config)
        elif config["benchmark"] == "ClinDiag":
            self.benchmark = _ClinDiag(config)
        else:
            raise ValueError(f"Unsupported benchmark: {config['benchmark']}")

    def __len__(self) -> int:
        """Len."""

        return len(self.benchmark)

    def reset(self) -> dict:
        """Reset."""

        return self.benchmark.reset()

    def step(self, action: Any) -> dict:
        """Step."""

        return self.benchmark.step(action)

    def skip(self) -> dict:
        """Advance one known completed case without unnecessary setup."""

        skip = getattr(self.benchmark, "skip", None)
        return skip() if skip is not None else self.benchmark.reset()

    def finalize_case(self, success: bool) -> None:
        """Finalize benchmark-local resources after durable result output."""

        finalize = getattr(self.benchmark, "finalize_case", None)
        if finalize is not None:
            finalize(success)

    def note_turn(self, turn: int, max_turns: int) -> None:
        """Forward the harness interaction-turn counter to the benchmark."""

        note = getattr(self.benchmark, "note_turn", None)
        if note is not None:
            note(turn, max_turns)

    def attach_logger(self, logger) -> None:
        """Forward the per-case event logger to benchmarks that use it."""

        attach = getattr(self.benchmark, "attach_logger", None)
        if attach is not None:
            attach(logger)

    def close(self) -> None:
        """Close."""

        self.benchmark.close()


def _tool_message_reports_failure(content: object) -> bool:
    """Recognize generic MCP failure envelopes without tool-name rules."""

    text = str(content).strip()
    lowered = text.casefold()
    if lowered.startswith("[tool execution error:"):
        return True
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict):
        status = str(payload.get("status", "")).strip().casefold()
        if status in {"error", "failed", "failure"}:
            return True
        error = payload.get("error")
        return error not in (None, "", False)
    if lowered.startswith("root(") or len(text) > 512:
        return False
    return bool(
        re.search(
            r"(?:^|\b)(?:not found|does not exist|failed|failure|error|"
            r"invalid|denied|forbidden|unavailable)(?:\b|$)",
            lowered,
        )
    )


class _ALFWorld:
    """Adapt the ALFWorld text environment to the shared agent interface."""

    def __init__(self, config: dict):
        """Initialize configuration and episode-local runtime state."""

        from alfworld.agents.environment import get_environment

        self.data = ROOT / "benchmarks" / "ALFWorld"
        environment = get_environment("AlfredTWEnv")(
            self._environment_config(config),
            train_eval=config["split"],
        )

        games = sorted(environment.game_files)
        environment.game_files = games[config["start_case"] : config["end_case"]]
        environment.num_games = len(environment.game_files)

        self.environment = environment.init_env(batch_size=1)
        self.total = len(environment.game_files)
        self.case_id = ""
        self.task = ""
        self.score = 0.0

    def __len__(self) -> int:
        """Len."""

        return self.total

    def reset(self) -> dict:
        """Reset."""

        observations, infos = self.environment.reset()
        game_file = Path(infos["extra.gamefile"][0])

        self.case_id = str(game_file.parent.relative_to(self.data))
        self.task = self._task_from_observation(observations[0])
        self.score = 0.0

        return self._state(observations[0], infos, done=False)

    def step(self, action: str) -> dict:
        """Step."""

        observations, scores, dones, infos = self.environment.step([action])
        self.score = float(scores[0])
        return self._state(observations[0], infos, done=bool(dones[0]))

    def close(self) -> None:
        """Close."""

        self.environment.close()

    def _state(self, observation: str, infos: dict, done: bool) -> dict:
        """State."""

        return {
            "case_id": self.case_id,
            "task": self.task,
            "observation": observation,
            "actions": list(infos["admissible_commands"][0]),
            "done": done,
            "success": bool(infos["won"][0]),
            "score": self.score,
        }

    def _environment_config(self, config: dict) -> dict:
        """Environment config."""

        data = self.data
        return {
            "dataset": {
                "data_path": str(data / "json_2.1.1" / "train"),
                "eval_id_data_path": str(data / "json_2.1.1" / "valid_seen"),
                "eval_ood_data_path": str(data / "json_2.1.1" / "valid_unseen"),
                "num_train_games": -1,
                "num_eval_games": -1,
            },
            "logic": {
                "domain": str(data / "logic" / "alfred.pddl"),
                "grammar": str(data / "logic" / "alfred.twl2"),
            },
            "env": {
                "goal_desc_human_anns_prob": 0.0,
                "domain_randomization": False,
                "task_types": config["task_types"],
                "expert_type": "handcoded",
            },
            "general": {"training_method": "dagger"},
            "dagger": {"training": {"max_nb_steps_per_episode": config["max_steps"]}},
        }

    @staticmethod
    def _task_from_observation(observation: str) -> str:
        """Task from observation."""

        marker = "Your task is to:"
        return observation.split(marker, maxsplit=1)[1].strip()


class _LOCABench:
    """Adapt official LOCA tasks and MCP tools to the shared agent interface."""

    def __init__(self, config: dict):
        """Initialize configuration and episode-local runtime state."""

        self.config = config
        self.data = ROOT / "benchmarks" / "LOCA-Bench"

        self.npm_cache = (
            Path(config.get("loca_npm_cache", self._default_npm_cache()))
            .expanduser()
            .resolve()
        )
        retention = config.get("loca_workspace_retention", "failed")
        if retention not in {"all", "failed", "none"}:
            raise ValueError("loca_workspace_retention must be all, failed, or none")
        self.workspace_retention = retention

        if str(self.data) not in sys.path:
            sys.path.insert(0, str(self.data))
        os.environ["PYTHONPATH"] = self._loca_pythonpath(self.data)

        length = config["loca_environment_length"]
        task_config = self.data / config.get(
            "loca_task_config",
            f"task-configs/final_{length}_set_config.json",
        )
        with task_config.open(encoding="utf-8") as file:
            cases = json.load(file)["configurations"]

        task_names = config.get("loca_task_names", "all")
        if task_names != "all":
            cases = [case for case in cases if case["name"] in task_names]

        seeds = config.get("loca_seeds")
        if seeds is not None:
            cases = [case for case in cases if case["env_params"]["seed"] in seeds]

        self.cases = cases[config["start_case"] : config["end_case"]]
        self.total = len(self.cases)
        self.cursor = 0
        self.case_id = ""
        self.task = ""
        self.actions: list[dict] = []
        self.score = 0.0
        self.environment = None
        self.tool = None
        self.workspace: Path | None = None
        self.agent_workspace: Path | None = None
        self.artifact_dir: Path | None = None
        self.artifact_contracts: dict[str, dict[str, Any]] = {}
        self._workspace_file_index: dict[str, tuple[int, int]] = {}
        self.expected_mcp_servers: set[str] = set()
        self.last_info: dict = {}

    def __len__(self) -> int:
        """Len."""

        return self.total

    def reset(self) -> dict:
        """Reset."""

        self.close()
        case = self.cases[self.cursor]
        self.cursor += 1

        task_name = case["name"]
        seed = case["env_params"]["seed"]
        length = self.config["loca_environment_length"]
        self.case_id = f"{length}/{task_name}/seed_{seed}"
        self.score = 0.0

        workspace_name = (
            f"{self.config['method']}_{length}_{task_name}_"
            f"seed_{seed}_pid_{os.getpid()}"
        )
        self.workspace = self.data / ".pos_workspaces" / workspace_name
        local_db = self.workspace / "local_db"
        agent_workspace = self.workspace / "agent_workspace"
        self.agent_workspace = agent_workspace
        self.artifact_dir = self.workspace / ".pos_artifacts" / "tool_results"
        self.artifact_contracts = {}
        local_db.mkdir(parents=True, exist_ok=True)
        agent_workspace.mkdir(parents=True, exist_ok=True)

        self._start_environment(case, agent_workspace)
        with self._suppress_output():
            _, info, user_prompt, tools = self.environment.reset()
        # LOCA environments may recreate the whole task directory during reset,
        # so runner-owned artifact directories must be created afterwards.
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._workspace_file_index = self._snapshot_workspace_files()
        self.task = user_prompt
        self.actions = tools[0] if tools else []
        attach_loca_capabilities(self.actions)
        self._validate_actions()
        self.last_info = info
        self._last_tool_success_counter = int(info.get("tool_success_counter", 0))
        self.last_execution: dict[str, Any] | None = None

        return self._state(user_prompt, done=False)

    def skip(self) -> dict:
        """Advance over an existing result without constructing its sandbox."""

        self.close()
        case = self.cases[self.cursor]
        self.cursor += 1
        task_name = case["name"]
        seed = case["env_params"]["seed"]
        length = self.config["loca_environment_length"]
        self.case_id = f"{length}/{task_name}/seed_{seed}"
        return {"case_id": self.case_id}

    def step(self, action: dict[str, Any]) -> dict:
        """Step."""

        before_files = self._workspace_file_index
        previous_success_counter = self._last_tool_success_counter
        tool_response = self._tool_response(action)
        observation, reward, terminated, truncated, info = self.environment.step_openai(
            tool_response, verbose=False
        )
        self.score = float(reward)
        self.last_info = info
        current_success_counter = int(info.get("tool_success_counter", 0))
        self.last_execution = self._execution_record(
            action,
            self._tool_execution_succeeded(
                observation,
                counter_advanced=current_success_counter > previous_success_counter,
            ),
        )
        self._last_tool_success_counter = current_success_counter

        if self.last_execution.get("success") and self.agent_workspace is not None:
            contract = capture_excel_artifact_contract(
                action,
                agent_workspace=self.agent_workspace,
            )
            if contract is not None:
                self.artifact_contracts.setdefault(contract["path"], contract)

        after_files = self._snapshot_workspace_files()
        producer = str(action.get("name", "unknown_tool"))
        outputs = self._workspace_output_delta(
            before_files,
            after_files,
            producer=producer,
        )
        validation_output = validated_workspace_output(
            observation,
            agent_workspace=self.agent_workspace,
            producer=producer,
            artifact_contracts=self.artifact_contracts,
        )
        if validation_output is not None:
            by_path = {str(item.get("path")): item for item in outputs}
            existing = by_path.get(validation_output["path"])
            if existing is None:
                outputs.append(validation_output)
            else:
                existing.update(validation_output)
        self._workspace_file_index = after_files
        observation = self._materialize_observation(observation, action)
        observation = append_workspace_output_manifest(
            observation,
            outputs=outputs,
        )
        if (
            validation_output is not None
            and validation_output.get("validation_status") == "failed"
        ):
            summary = json.loads(str(validation_output.get("validation_summary", "{}")))
            failed_checks = sorted(
                name
                for name, check in summary.items()
                if not check.get("passed", False)
            )
            observation += (
                "\nValidation failed: runner checks did not pass: "
                + ", ".join(failed_checks)
                + "."
            )
        return self._state(
            observation,
            done=bool(terminated or truncated),
        )

    def close(self) -> None:
        """Close."""

        if self.tool is not None:
            self.tool.close()
            self.tool = None
        self.environment = None

    def finalize_case(self, success: bool) -> None:
        """Close the case and retain only workspaces selected by policy."""

        self.close()
        should_delete = self.workspace_retention == "none" or (
            self.workspace_retention == "failed" and success
        )
        if should_delete:
            self._delete_current_workspace()

    def _delete_current_workspace(self) -> None:
        """Delete exactly one resolved child of the managed workspace root."""

        if self.workspace is None or not self.workspace.exists():
            return
        root = (self.data / ".pos_workspaces").resolve()
        workspace = self.workspace.resolve()
        if workspace == root or workspace.parent != root:
            raise RuntimeError(
                f"Refusing to delete unsafe LOCA workspace path: {workspace}"
            )
        shutil.rmtree(workspace)

    @staticmethod
    def _tool_execution_succeeded(
        observation: str,
        *,
        counter_advanced: bool,
    ) -> bool:
        """Interpret the ToolEnvWrapperOpenAI protocol, not tool prose."""

        if counter_advanced:
            return True
        try:
            messages = json.loads(observation)
        except (TypeError, json.JSONDecodeError):
            return False
        tool_messages = (
            [
                item
                for item in messages
                if isinstance(item, dict) and item.get("role") == "tool"
            ]
            if isinstance(messages, list)
            else []
        )
        if not tool_messages:
            return False
        return all(
            not _tool_message_reports_failure(item.get("content", ""))
            for item in tool_messages
        )

    def _execution_record(
        self, action: dict[str, Any], success: bool
    ) -> dict[str, Any]:
        """Expose one adapter-confirmed explicitly declared capability."""
        name = str(action.get("name", ""))
        capability = None
        for available in self.actions:
            function = available.get("function", {})
            if isinstance(function, dict) and function.get("name") == name:
                declared = function.get("x_pos_capability")
                capability = deepcopy(declared) if isinstance(declared, dict) else None
                break
        return {
            "action": name,
            "arguments": deepcopy(action.get("arguments", {})),
            "success": bool(success),
            "capability": capability,
        }

    def _materialize_observation(
        self,
        observation: str,
        action: dict[str, Any],
    ) -> str:
        """Replace only large structured LOCA results with artifact references."""

        settings = self.config.get("loca_observation_artifacts", {})
        if not settings.get("enabled", False):
            return observation
        if not self._has_python_execute_action():
            return observation
        if self.artifact_dir is None:
            return observation

        capability = (
            self.last_execution.get("capability")
            if isinstance(self.last_execution, dict)
            else None
        )
        transform_stdout = bool(
            isinstance(capability, dict)
            and capability.get("effect") == "transform"
            and settings.get("materialize_transform_stdout", True)
        )
        return materialize_loca_observation(
            observation,
            artifact_dir=self.artifact_dir,
            agent_artifact_dir=str(self.artifact_dir),
            case_id=self.case_id,
            action=action,
            tool_use_counter=int(self.last_info.get("tool_use_counter", 0)),
            inline_max_chars=int(settings.get("inline_max_chars", 8000)),
            preview_items=int(settings.get("preview_items", 3)),
            structured_only=(
                bool(settings.get("structured_only", True)) and not transform_stdout
            ),
        )

    def _snapshot_workspace_files(self) -> dict[str, tuple[int, int]]:
        """Return safe metadata for durable files visible to the task agent."""

        if self.agent_workspace is None or not self.agent_workspace.is_dir():
            return {}
        root = self.agent_workspace.resolve()
        snapshot: dict[str, tuple[int, int]] = {}
        for candidate in root.rglob("*"):
            try:
                relative = candidate.relative_to(root)
                if relative.parts and relative.parts[0] in {
                    ".python_tmp",
                    "__pycache__",
                }:
                    continue
                resolved = candidate.resolve()
                resolved.relative_to(root)
                if not resolved.is_file():
                    continue
                stat = resolved.stat()
            except (OSError, ValueError):
                continue
            snapshot[str(relative)] = (stat.st_size, stat.st_mtime_ns)
        return snapshot

    def _workspace_output_delta(
        self,
        before: dict[str, tuple[int, int]],
        after: dict[str, tuple[int, int]],
        *,
        producer: str,
    ) -> list[dict[str, Any]]:
        """Describe only files created or modified by the just-finished action."""

        if self.agent_workspace is None:
            return []
        root = self.agent_workspace.resolve()
        outputs: list[dict[str, Any]] = []
        for relative in sorted(after):
            metadata = after[relative]
            previous = before.get(relative)
            if previous == metadata:
                continue
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                continue
            outputs.append(
                {
                    "path": str(path),
                    "bytes": metadata[0],
                    "change": "created" if previous is None else "modified",
                    "producer": producer,
                }
            )
        return outputs

    def _has_python_execute_action(self) -> bool:
        """Return whether this case can consume runner-owned artifacts."""

        names = {action.get("function", {}).get("name", "") for action in self.actions}
        return any(
            name == "python_execute" or name.endswith("_python_execute")
            for name in names
        )

    def _start_environment(
        self,
        case: dict,
        agent_workspace: Path,
    ) -> None:
        """Start environment."""

        from gem.tools.mcp_tool import MCPTool
        from gem.tools.tool_env_wrapper import ToolEnvWrapperOpenAI

        environment_class = self._environment_class(case["env_class"])
        environment_params = deepcopy(case["env_params"])
        environment_params.setdefault("task_dir", str(self.workspace))

        with self._suppress_output():
            environment = environment_class(**environment_params)

        mcp_config = self._mcp_config(
            deepcopy(case["mcp_servers"]),
            self.workspace,
            agent_workspace,
            npm_cache=self.npm_cache,
        )
        self.expected_mcp_servers = set(mcp_config["mcpServers"])
        self.tool = MCPTool(
            mcp_config,
            validate_on_init=False,
            execution_timeout=float(self.config.get("loca_tool_timeout_seconds", 120)),
            fix_schema_for_openai=False,
        )
        self.environment = ToolEnvWrapperOpenAI(
            environment,
            tools=[self.tool],
            max_tool_uses=self.config.get("loca_max_tool_uses", 500),
        )

    @staticmethod
    def _environment_class(class_path: str):
        """Environment class."""

        module_path, class_name = class_path.rsplit(".", maxsplit=1)
        return getattr(import_module(module_path), class_name)

    @staticmethod
    def _mcp_config(
        server_configs: dict,
        task_workspace: Path,
        agent_workspace: Path,
        npm_cache: Path | None = None,
    ) -> dict:
        """Resolve MCP server workspaces and explicitly pass subprocess cache settings."""

        from gem.tools.mcp_server.config_loader import build_server_config

        mcp_servers = {}
        for server_name, server in server_configs.items():
            if not server.get("enabled", True):
                continue

            parameters = server.get("params", {})
            for key, value in parameters.items():
                if isinstance(value, str):
                    parameters[key] = value.replace(
                        "{task_workspace}",
                        str(task_workspace),
                    ).replace(
                        "{agent_workspace}",
                        str(agent_workspace),
                    )
            parameters["task_workspace"] = str(task_workspace)
            parameters["agent_workspace"] = str(agent_workspace)
            server_config = build_server_config(
                server_type=server["type"],
                params=parameters,
                server_name=server_name,
            )
            process_path = _LOCABench._mcp_process_path()
            for definition in server_config.values():
                environment = dict(definition.get("env", {}))
                environment["PATH"] = process_path
                if Path(str(definition.get("command", ""))).name == "npx":
                    cache_path = (
                        (npm_cache or _LOCABench._default_npm_cache())
                        .expanduser()
                        .resolve()
                    )
                    cache_path.mkdir(parents=True, exist_ok=True)

                    environment["npm_config_cache"] = str(cache_path)
                environment.setdefault(
                    "LOCA_QUIET",
                    os.environ.get("LOCA_QUIET", "1"),
                )
                environment.setdefault("PYTHONUNBUFFERED", "1")
                definition["env"] = environment
            mcp_servers.update(server_config)
        return {"mcpServers": mcp_servers}

    @staticmethod
    def _default_npm_cache() -> Path:
        """Return a portable per-user npm cache directory."""

        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        return cache_root / "pos" / "npm"

    @staticmethod
    def _mcp_process_path() -> str:
        """Expose the active Python environment and available Node tools to MCP children."""

        entries = [str(Path(sys.executable).resolve().parent)]
        user_tool_bin = Path.home() / ".local" / "bin"
        if user_tool_bin.is_dir():
            entries.append(str(user_tool_bin))
        entries.extend(
            entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry
        )

        return os.pathsep.join(dict.fromkeys(entries))

    @staticmethod
    def _loca_pythonpath(loca_root: Path) -> str:
        """Loca pythonpath."""

        entries = [str(loca_root.resolve())]
        entries.extend(
            entry
            for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if entry
        )

        return os.pathsep.join(dict.fromkeys(entries))

    def _validate_actions(self) -> None:
        """Validate actions."""

        if not self.actions:
            raise RuntimeError(
                f"LOCA-Bench case {self.case_id} exposed no MCP tools. "
                "Check the MCP subprocess PATH and server startup logs."
            )
        names = {action.get("function", {}).get("name", "") for action in self.actions}
        missing_servers = sorted(
            server_name
            for server_name in self.expected_mcp_servers
            if not any(name.startswith(f"{server_name}_") for name in names)
        )
        if missing_servers:
            raise RuntimeError(
                f"LOCA-Bench case {self.case_id} is missing tools from enabled "
                f"MCP servers: {', '.join(missing_servers)}."
            )
        if not any(
            name == "claim_done" or name.endswith("_claim_done") for name in names
        ):
            raise RuntimeError(
                f"LOCA-Bench case {self.case_id} has no claim_done tool; "
                "the benchmark cannot evaluate task completion."
            )

    @staticmethod
    @contextmanager
    def _suppress_output():
        """Suppress output."""

        with open(os.devnull, "w", encoding="utf-8") as null_output:
            with redirect_stdout(null_output), redirect_stderr(null_output):
                yield

    def _tool_response(self, action: dict[str, Any]) -> dict:
        """Tool response."""

        tool_call = {
            "id": (f"call_{self.cursor}_{self.last_info.get('tool_use_counter', 0)}"),
            "type": "function",
            "function": {
                "name": action["name"],
                "arguments": json.dumps(
                    action.get("arguments", {}),
                    ensure_ascii=False,
                ),
            },
        }
        return {
            "type": "tool",
            "data": [tool_call],
            "call_messages": {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            },
        }

    def _state(self, observation: str, done: bool) -> dict:
        """State."""

        success = done and self.score >= 1.0
        return {
            "case_id": self.case_id,
            "task": self.task,
            "observation": observation,
            "actions": self.actions,
            "done": done,
            "success": success,
            "score": self.score,
            "benchmark_info": {
                "environment_length": self.config["loca_environment_length"],
                "tool_use_counter": self.last_info.get(
                    "tool_use_counter",
                    0,
                ),
                "tool_success_counter": self.last_info.get(
                    "tool_success_counter",
                    0,
                ),
                "workspace": str(self.workspace),
                "loca_commit": LOCA_BENCH_COMMIT,
                "last_tool_execution": deepcopy(self.last_execution),
            },
        }


class _RCA100:
    """Adapt bounded observability queries and evaluator-only RCA labels."""

    def __init__(self, config: dict):
        """Initialize configuration and episode-local runtime state."""

        from benchmarks.rca100_env import RCA100ToolRegistry

        self.config = config
        self.data = ROOT / "benchmarks" / "RCA100"
        self.answer_key_dir = self.data / "answer_key"

        manifest = [
            line.strip()
            for line in (self.data / "manifest.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        task_ids = config.get("rca_task_ids", "all")
        if task_ids != "all":
            selected = set(task_ids)
            manifest = [task_id for task_id in manifest if task_id in selected]

        self.cases = manifest[config["start_case"] : config["end_case"]]
        self.total = len(self.cases)
        self.cursor = 0

        tools_module = config.get(
            "rca_tools_module",
            "benchmarks/RCA100/tools.py",
        )
        tools_path = Path(tools_module)
        if not tools_path.is_absolute():
            tools_path = ROOT / tools_path
        self.registry = RCA100ToolRegistry(tools_path)
        self.observation_max_chars = int(config.get("rca_observation_max_chars", 20000))

        self.case = None
        self.case_id = ""
        self.task = ""
        self.actions: list[dict] = []
        self.tool_use_counter = 0
        self.current_turn = 0
        self.max_turns = int(config["max_steps"])
        self.verdict: dict | None = None

    def __len__(self) -> int:
        """Len."""

        return self.total

    def reset(self) -> dict:
        """Reset."""

        from benchmarks.rca100_env import RCA100Case

        self.close()
        task_id = self.cases[self.cursor]
        self.cursor += 1

        self.case = RCA100Case(self.data / "cases" / task_id)
        self.case_id = task_id
        self.tool_use_counter = 0
        self.current_turn = 0
        self.verdict = None

        task_input = {
            key: value for key, value in self.case.task.items() if key != "prompt_text"
        }
        self.task = json.dumps(task_input, ensure_ascii=False, indent=2)
        self.actions = self.registry.schemas()

        observation = (
            "The incident observability data (metrics, logs, traces, "
            "events, alerts, topology) is ready but not included inline; "
            "read it through the available tools. The alert entity in the "
            "task is only the symptom entry point and is frequently not "
            "the root cause. Investigate the data, then call "
            "submit_root_cause exactly once with your conclusion."
        )
        return self._state(observation, done=False)

    def skip(self) -> dict:
        """Advance over an existing result without loading its data."""

        self.close()
        task_id = self.cases[self.cursor]
        self.cursor += 1
        self.case_id = task_id
        return {"case_id": task_id}

    def note_turn(self, turn: int, max_turns: int) -> None:
        """Note turn."""

        self.current_turn = turn + 1
        self.max_turns = max_turns

    def step(self, action: Any) -> dict:
        """Step."""

        from benchmarks.rca100_env import (
            SUBMIT_TOOL_NAME,
            cap_observation,
            score_submission,
            validate_submission,
        )

        problem = self._action_problem(action)
        if problem is not None:
            return self._state(f"Invalid action: {problem}", done=False)

        name = action["name"]
        arguments = action.get("arguments") or {}
        self.tool_use_counter += 1

        if name == SUBMIT_TOOL_NAME:
            problems = validate_submission(arguments)
            if problems:
                observation = (
                    "Submission rejected: "
                    + "; ".join(problems)
                    + ". Fix the arguments and call submit_root_cause again."
                )
                return self._state(observation, done=False)

            self.verdict = score_submission(
                arguments,
                self.answer_key_dir,
                self.case_id,
            )
            observation = (
                "Answer received: "
                + json.dumps(
                    {
                        "root_cause_entity": arguments["root_cause_entity"],
                        "fault_type": self.verdict["submitted_fault_type"],
                    },
                    ensure_ascii=False,
                )
                + ". The case is finished."
            )
            return self._state(observation, done=True)

        if not self.registry.has_tool(name):
            available = ", ".join(schema["function"]["name"] for schema in self.actions)
            observation = (
                f"Unknown tool: {name}. The only available tools are: {available}."
            )
            return self._state(observation, done=False)

        observation = cap_observation(
            self.registry.dispatch(name, arguments, self.case),
            self.observation_max_chars,
        )
        return self._state(observation, done=False)

    def close(self) -> None:
        """Close."""

        if self.case is not None:
            self.case.release()
            self.case = None

    def finalize_case(self, success: bool) -> None:
        """Finalize case."""

        self.close()

    @staticmethod
    def _action_problem(action: Any) -> str | None:
        """Action problem."""

        if not isinstance(action, dict) or not isinstance(action.get("name"), str):
            return (
                'the action must be {"name": "tool name", "arguments": {}} '
                "with a tool from the available tools list"
            )
        arguments = action.get("arguments")
        if arguments is not None and not isinstance(arguments, dict):
            return "arguments must be a JSON object"
        return None

    def _state(self, observation: str, done: bool) -> dict:
        """State."""

        verdict = self.verdict
        return {
            "case_id": self.case_id,
            "task": self.task,
            "observation": observation,
            "actions": self.actions,
            "done": done,
            "success": bool(verdict["success"]) if verdict else False,
            "score": float(verdict["score"]) if verdict else 0.0,
            "benchmark_info": {
                "task_version": self.case.task.get("task_version")
                if self.case is not None
                else None,
                "tool_use_counter": self.tool_use_counter,
                "no_submission": verdict is None,
                "entity_correct": bool(verdict["entity_correct"]) if verdict else False,
                "type_correct": bool(verdict["type_correct"]) if verdict else False,
                "scoring": verdict,
            },
        }


class _ClinDiag:
    """Adapt patient information requests and evaluator-only diagnosis labels."""

    def __init__(self, config: dict):
        """Initialize configuration and episode-local runtime state."""

        from benchmarks.clindiag_env import (
            BenchmarkModelCaller,
            provider_tool_schemas,
        )

        self.config = config
        self.data = ROOT / "benchmarks" / "ClinDiag"
        self.answer_key_dir = self.data / "answer_key"

        manifest = [
            line.strip()
            for line in (self.data / "manifest.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        case_ids = config.get("clindiag_case_ids", "all")
        if case_ids != "all":
            selected = set(case_ids)
            manifest = [case for case in manifest if case in selected]
        subsets = config.get("clindiag_subsets", "all")
        if subsets != "all":
            wanted = set(subsets)
            manifest = [
                case for case in manifest if case.split("/", maxsplit=1)[0] in wanted
            ]

        self.cases = manifest[config["start_case"] : config["end_case"]]
        self.total = len(self.cases)
        self.cursor = 0

        self.call_provider = BenchmarkModelCaller(config["clindiag_provider_model"])
        self.call_judge = BenchmarkModelCaller(config["clindiag_judge_model"])
        self.judge_max_retries = int(
            config["clindiag_judge_model"].get("max_json_retries", 2)
        )
        self.tool_schemas = provider_tool_schemas()
        self.observation_max_chars = int(
            config.get("clindiag_observation_max_chars", 8000)
        )

        self.case = None
        self.case_slug = ""
        self.subset = ""
        self.case_id = ""
        self.task = ""
        self.actions: list[dict] = []
        self.tool_use_counter = 0
        self.current_turn = 0
        self.max_turns = int(config["max_steps"])
        self.verdict: dict | None = None
        self._token_usage_before: dict | None = None
        self.logger = None

    def __len__(self) -> int:
        """Len."""

        return self.total

    def reset(self) -> dict:
        """Reset."""

        from benchmarks.clindiag_env import ClinDiagCase

        self.close()
        case_ref = self.cases[self.cursor]
        self.cursor += 1

        subset, upstream_id = case_ref.split("/", maxsplit=1)
        slug = f"{subset}__{upstream_id}"
        self.case = ClinDiagCase(self.data / "cases" / slug)
        self.case_id = case_ref
        self.case_slug = slug
        self.subset = subset
        self.tool_use_counter = 0
        self.current_turn = 0
        self.verdict = None
        self._token_usage_before = self._benchmark_token_usage()

        self.task = json.dumps(
            {
                "case_id": case_ref,
                "subset": subset,
                "initial_presentation": self.case.initial_information,
            },
            ensure_ascii=False,
            indent=2,
        )
        self.actions = list(self.tool_schemas)

        observation = (
            "The patient is in front of you; only the initial presentation "
            "is known so far. The full record (history, physical "
            "examination, diagnostic tests) must be obtained through the "
            "tools, one specific question per turn. Investigate, then call "
            "submit_diagnosis exactly once with your conclusion."
        )
        return self._state(observation, done=False)

    def skip(self) -> dict:
        """Advance over an existing result without loading its data."""

        self.close()
        case_ref = self.cases[self.cursor]
        self.cursor += 1
        self.case_id = case_ref
        return {"case_id": case_ref}

    def note_turn(self, turn: int, max_turns: int) -> None:
        """Note turn."""

        self.current_turn = turn + 1
        self.max_turns = max_turns

    def attach_logger(self, logger) -> None:
        """Attach logger."""

        self.logger = logger

    def step(self, action: Any) -> dict:
        """Step."""

        from benchmarks.clindiag_env import (
            QUERY_TOOLS,
            SUBMIT_TOOL_NAME,
            answer_key_access_problem,
            ask_provider,
            cap_observation,
            judge_submission,
            load_ground_truth,
            validate_submission,
        )

        problem = self._action_problem(action)
        if problem is not None:
            return self._state(f"Invalid action: {problem}", done=False)

        name = action["name"]
        arguments = action.get("arguments") or {}
        self.tool_use_counter += 1

        if name == SUBMIT_TOOL_NAME:
            problems = validate_submission(arguments)
            if problems:
                observation = (
                    "Submission rejected: "
                    + "; ".join(problems)
                    + ". Fix the arguments and call submit_diagnosis again."
                )
                return self._state(observation, done=False)

            self.verdict = judge_submission(
                self.call_judge,
                arguments,
                load_ground_truth(self.answer_key_dir, self.case_slug),
                max_retries=self.judge_max_retries,
                logger=self._event_logger(),
                case_id=self.case_id,
                step=self.current_turn,
            )
            observation = (
                "Diagnosis received: "
                + json.dumps(
                    {
                        "final_diagnosis": arguments["final_diagnosis"],
                    },
                    ensure_ascii=False,
                )
                + ". The case is finished."
            )
            return self._state(observation, done=True)

        if name not in QUERY_TOOLS:
            available = ", ".join(schema["function"]["name"] for schema in self.actions)
            observation = (
                f"Unknown tool: {name}. The only available tools are: {available}."
            )
            return self._state(observation, done=False)

        guard_problem = answer_key_access_problem(arguments)
        if guard_problem is not None:
            return self._state(
                f"Tool call rejected: {guard_problem}",
                done=False,
            )
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            return self._state(
                f"Tool call error: {name} requires one non-empty "
                "'question' string argument.",
                done=False,
            )

        answer = ask_provider(
            self.call_provider,
            name,
            question,
            self.case,
        )
        self._event_logger().log(
            "clindiag_provider_exchange",
            case_id=self.case_id,
            step=self.current_turn,
            tool=name,
            question=question,
            answer=answer,
        )
        observation = cap_observation(answer, self.observation_max_chars)
        return self._state(observation, done=False)

    def close(self) -> None:
        """Close."""

        if self.case is not None:
            self.case.release()
            self.case = None

    def finalize_case(self, success: bool) -> None:
        """Finalize case."""

        self.close()

    def _event_logger(self):
        """Event logger."""

        if self.logger is not None:
            return self.logger

        class _NullLogger:
            def log(self, event, **data):
                pass

        return _NullLogger()

    def _benchmark_token_usage(self) -> dict:
        """Benchmark token usage."""

        return {
            "provider": self.call_provider.token_usage(),
            "judge": self.call_judge.token_usage(),
        }

    def _case_token_usage(self) -> dict:
        """Case token usage."""

        current = self._benchmark_token_usage()
        before = self._token_usage_before or {
            role: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            for role in current
        }
        return {
            role: {
                field: current[role][field] - before[role][field]
                for field in current[role]
            }
            for role in current
        }

    @staticmethod
    def _action_problem(action: Any) -> str | None:
        """Action problem."""

        if not isinstance(action, dict) or not isinstance(action.get("name"), str):
            return (
                'the action must be {"name": "tool name", "arguments": {}} '
                "with a tool from the available tools list"
            )
        arguments = action.get("arguments")
        if arguments is not None and not isinstance(arguments, dict):
            return "arguments must be a JSON object"
        return None

    def _state(self, observation: str, done: bool) -> dict:
        """State."""

        verdict = self.verdict
        return {
            "case_id": self.case_id,
            "task": self.task,
            "observation": observation,
            "actions": self.actions,
            "done": done,
            "success": bool(verdict["success"]) if verdict else False,
            "score": float(verdict["score"]) if verdict else 0.0,
            "benchmark_info": {
                "subset": getattr(self, "subset", None),
                "tool_use_counter": self.tool_use_counter,
                "no_submission": verdict is None,
                "diagnosis_correct": bool(verdict["diagnosis_correct"])
                if verdict
                else False,
                "differential_hit": bool(verdict["differential_hit"])
                if verdict
                else False,
                "scoring": verdict,
                "token_usage": self._case_token_usage(),
            },
        }
