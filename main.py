"""Run PoS and Raw/ReAct experiments through a shared benchmark interface."""

from __future__ import annotations
import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
import yaml
from openai import OpenAI
from tqdm import tqdm
from baselines.react import ReAct
from belief.manager import BeliefManager
from benchmarks.adapter import Adapter
from contexts.raw import RawTrajectoryContext
from utils.logger import Logger
from utils.public_function import (
    call_with_backoff,
    render_case_index_markdown,
    render_events_markdown,
)

ROOT = Path(__file__).resolve().parent
WORKER_ENV = "POS_EXPERIMENT_WORKER"


class ModelCaller:
    """OpenAI-compatible model client with cumulative provider-reported token usage."""

    def __init__(self, config: dict):
        """Initialize configuration and episode-local runtime state."""
        self.config = config
        self.client = OpenAI(
            api_key=os.getenv(config["api_key_env"], "EMPTY"),
            base_url=config["base_url"],
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def __call__(self, messages: list[dict]) -> str:
        """Execute the configured operation and return its result."""
        response = call_with_backoff(
            lambda: self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                extra_body=self.config.get("extra_body", {}),
            ),
            f"{self.config['model']} chat",
        )
        usage = response.usage
        self.input_tokens += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        return response.choices[0].message.content

    def token_usage(self) -> dict[str, int]:
        """Return cumulative input, output, and total token counts."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


def load_config(path: str) -> dict:
    """Load one experiment YAML without implicit inheritance or overrides."""
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_model(config: dict) -> ModelCaller:
    """Create a model caller with cumulative usage accounting."""
    return ModelCaller(config)


def collect_token_usage(
    call_model,
    call_compressor=None,
    call_belief=None,
    call_sentinel_consistency=None,
    call_sentinel_progress=None,
) -> dict:
    """Collect cumulative role usage without double-counting the Sentinel subtotal."""
    fields = ("input_tokens", "output_tokens", "total_tokens")
    zero = {field: 0 for field in fields}
    calls = {
        "agent": call_model,
        "compressor": call_compressor,
        "belief": call_belief,
        "sentinel_consistency": call_sentinel_consistency,
        "sentinel_progress": call_sentinel_progress,
    }
    roles = {
        name: caller.token_usage()
        for name, caller in calls.items()
        if caller is not None
    }
    roles.setdefault("compressor", zero)
    usage = {field: sum((role[field] for role in roles.values())) for field in fields}
    usage.update(roles)
    if call_sentinel_consistency is not None or call_sentinel_progress is not None:
        usage["sentinel"] = {
            field: roles.get("sentinel_consistency", zero)[field]
            + roles.get("sentinel_progress", zero)[field]
            for field in fields
        }
    return usage


def subtract_token_usage(current: dict, previous: dict) -> dict:
    """Compute per-episode usage from cumulative snapshots."""
    fields = ("input_tokens", "output_tokens", "total_tokens")
    roles = [
        role
        for role in (
            "agent",
            "compressor",
            "belief",
            "sentinel_consistency",
            "sentinel_progress",
            "sentinel",
        )
        if role in current
    ]
    return {
        **{field: current[field] - previous[field] for field in fields},
        **{
            role: {
                field: current[role][field] - previous[role][field] for field in fields
            }
            for role in roles
        },
    }


def model_directory_name(config: dict) -> str:
    """Model directory name."""
    temperature = str(config["temperature"]).replace(".", "_")
    return f"{config['model'].replace('/', '_')}_t{temperature}"


def sentinel_model_config(sentinel_config: dict, lane: str) -> dict:
    """Resolve a lane-specific model, falling back to the shared Sentinel model."""
    return sentinel_config.get(f"{lane}_model") or sentinel_config["model"]


def sentinel_lane_enabled(sentinel_config: dict, lane: str) -> bool:
    """Combine the Sentinel master switch with the selected lane switch."""
    return bool(
        sentinel_config.get("enabled", False)
        and sentinel_config.get(f"{lane}_enabled", True)
    )


def validated_run_name(config: dict) -> str | None:
    """Validated run name."""
    output_run_name = config.get("output_run_name")
    if output_run_name is None:
        return None
    output_run_name = str(output_run_name)
    if (
        not output_run_name
        or output_run_name in {".", ".."}
        or Path(output_run_name).name != output_run_name
    ):
        raise ValueError("output_run_name must be one non-empty directory name")
    return output_run_name


def output_directory(config: dict) -> Path:
    """Organize output by benchmark, method, model, and optional run name."""
    llm = f"{model_directory_name(config)}_{config['max_steps']}steps"
    selected_context = context_type(config)
    if selected_context == "belief":
        context_model = config["context"]["model"]
        context_name = context_type(config)
        llm = f"{llm}__{context_name}_{model_directory_name(context_model)}"
        if context_name == "belief":
            agent_view = config["context"].get("agent_view", "text")
            if agent_view != "text":
                llm = f"{llm}__agent_view_{agent_view}"
            sentinel_config = config["context"].get("sentinel", {})
            if sentinel_config.get("enabled", False):
                if sentinel_lane_enabled(sentinel_config, "consistency"):
                    consistency_model = sentinel_model_config(
                        sentinel_config, "consistency"
                    )
                    llm = f"{llm}__sentinel_consistency_{model_directory_name(consistency_model)}"
                else:
                    llm = f"{llm}__sentinel_consistency_off"
                if config["context"].get("task_mode", "execution") == "execution":
                    if sentinel_lane_enabled(sentinel_config, "progress"):
                        progress_model = sentinel_model_config(
                            sentinel_config, "progress"
                        )
                        llm = f"{llm}__progress_{model_directory_name(progress_model)}"
                    else:
                        llm = f"{llm}__progress_off"
            if config["context"].get("trapping", {}).get("enabled", False):
                trapping = config["context"]["trapping"]
                epsilon = str(trapping.get("recurrence_epsilon", 0.15)).replace(
                    ".", "_"
                )
                threshold = str(trapping.get("health_threshold", 0.25)).replace(
                    ".", "_"
                )
                cycle = str(trapping.get("cycle_threshold", 0.75)).replace(".", "_")
                llm = f"{llm}__belief_health_k{trapping.get('window_size', 8)}_l{trapping.get('max_cycle_length', 4)}_e{epsilon}_h{threshold}_r{cycle}"
                if config["context"].get("recovery", {}).get("enabled", False):
                    llm = f"{llm}__recovery"
    if config["benchmark"] == "RCA100":
        actions_per_turn = int(config.get("max_actions_per_turn", 1))
        if actions_per_turn > 1:
            llm = f"{llm}__k{actions_per_turn}"
    benchmark_output = ROOT / "outputs" / config["benchmark"]
    output_run_name = validated_run_name(config)
    if config["benchmark"] == "LOCA-Bench":
        benchmark_output = benchmark_output / config["loca_environment_length"]
        if output_run_name is not None:
            return benchmark_output / output_run_name
        return benchmark_output / config["method"] / llm
    experiment_name = config.get("experiment_name")
    if experiment_name:
        benchmark_output = benchmark_output / experiment_name
    return benchmark_output / config["method"] / (output_run_name or llm)


def load_completed_case(adapter: Adapter, case_path: Path) -> dict:
    """Load completed case."""
    state = adapter.skip()
    result = json.loads((case_path / "result.json").read_text(encoding="utf-8"))
    if state["case_id"] != result["case_id"]:
        raise ValueError(
            f"Stored result does not match the current benchmark order: {case_path.name} records {result['case_id']}; current task is {state['case_id']}"
        )
    return result


def completed_case_seconds(result: dict, case_path: Path) -> float:
    """Completed case seconds."""
    if "case_seconds" in result:
        return float(result["case_seconds"])
    events = [
        json.loads(line)
        for line in (case_path / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    started_at = datetime.fromisoformat(events[0]["time"])
    finished_at = datetime.fromisoformat(events[-1]["time"])
    return (finished_at - started_at).total_seconds()


def aggregate_result_token_usage(results: list[dict]) -> dict:
    """Aggregate stored episode usage, including episodes loaded during resume."""
    fields = ("input_tokens", "output_tokens", "total_tokens")
    roles = [
        role
        for role in (
            "agent",
            "compressor",
            "belief",
            "sentinel_consistency",
            "sentinel_progress",
            "sentinel",
        )
        if any((role in result["token_usage"] for result in results))
    ]
    return {
        **{
            field: sum((result["token_usage"][field] for result in results))
            for field in fields
        },
        **{
            role: {
                field: sum(
                    (
                        result["token_usage"].get(role, {}).get(field, 0)
                        for result in results
                    )
                )
                for field in fields
            }
            for role in roles
        },
    }


def build_summary(
    config: dict,
    selected_context: str,
    compressor_config: dict | None,
    results: list[dict],
    case_times: list[float],
) -> dict:
    """Build summary."""
    successes = sum((result["success"] for result in results))
    summary = {
        "benchmark": config["benchmark"],
        "method": config["method"],
        "model": config["model"],
        "cases": len(results),
        "successes": successes,
        "success_rate": successes / len(results),
        "average_case_seconds": round(sum(case_times) / len(case_times), 3),
        "token_usage": aggregate_result_token_usage(results),
    }
    if "experiment_name" in config:
        summary["experiment_name"] = config["experiment_name"]
    if "output_run_name" in config:
        summary["output_run_name"] = config["output_run_name"]
    benchmark_results = [
        result["benchmark_info"] for result in results if "benchmark_info" in result
    ]
    if benchmark_results and config["benchmark"] == "ClinDiag":

        def _subset_accuracy(subset_name):
            subset_items = [
                item for item in benchmark_results if item.get("subset") == subset_name
            ]
            if not subset_items:
                return None
            return sum(
                (bool(item["diagnosis_correct"]) for item in subset_items)
            ) / len(subset_items)

        summary["benchmark_info"] = {
            "diagnosis_accuracy": sum(
                (bool(item["diagnosis_correct"]) for item in benchmark_results)
            )
            / len(benchmark_results),
            "diagnosis_accuracy_rare": _subset_accuracy("rare"),
            "diagnosis_accuracy_challenging": _subset_accuracy("challenging"),
            "differential_hit_rate": sum(
                (bool(item["differential_hit"]) for item in benchmark_results)
            )
            / len(benchmark_results),
            "no_submission_count": sum(
                (bool(item["no_submission"]) for item in benchmark_results)
            ),
            "total_tool_uses": sum(
                (item["tool_use_counter"] for item in benchmark_results)
            ),
            "average_tool_uses": round(
                sum((item["tool_use_counter"] for item in benchmark_results))
                / len(benchmark_results),
                3,
            ),
            "token_usage": {
                role: {
                    field: sum(
                        (item["token_usage"][role][field] for item in benchmark_results)
                    )
                    for field in ("input_tokens", "output_tokens", "total_tokens")
                }
                for role in ("provider", "judge")
            },
        }
    elif benchmark_results and config["benchmark"] == "RCA100":
        summary["benchmark_info"] = {
            "entity_accuracy": sum(
                (bool(item["entity_correct"]) for item in benchmark_results)
            )
            / len(benchmark_results),
            "type_accuracy": sum(
                (bool(item["type_correct"]) for item in benchmark_results)
            )
            / len(benchmark_results),
            "joint_accuracy": sum(
                (
                    bool(item["entity_correct"]) and bool(item["type_correct"])
                    for item in benchmark_results
                )
            )
            / len(benchmark_results),
            "no_submission_count": sum(
                (bool(item["no_submission"]) for item in benchmark_results)
            ),
            "total_tool_uses": sum(
                (item["tool_use_counter"] for item in benchmark_results)
            ),
            "average_tool_uses": round(
                sum((item["tool_use_counter"] for item in benchmark_results))
                / len(benchmark_results),
                3,
            ),
        }
    elif benchmark_results:
        summary["benchmark_info"] = {
            "environment_length": benchmark_results[0]["environment_length"],
            "loca_commit": benchmark_results[0].get("loca_commit"),
            "total_tool_uses": sum(
                (item["tool_use_counter"] for item in benchmark_results)
            ),
            "average_tool_uses": round(
                sum((item["tool_use_counter"] for item in benchmark_results))
                / len(benchmark_results),
                3,
            ),
            "total_successful_tool_uses": sum(
                (item["tool_success_counter"] for item in benchmark_results)
            ),
        }
    if selected_context == "belief":
        summary["belief_model"] = config["context"]["model"]["model"]
        summary["belief_update_every_steps"] = config["context"].get(
            "update_every_steps", 1
        )
        summary["belief_task_mode"] = config["context"].get("task_mode", "execution")
        summary["belief_diagnostic_progress_threshold"] = config["context"].get(
            "diagnostic_progress_threshold", 0.3
        )
        summary["belief_trapping"] = config["context"].get("trapping", {})
        summary["belief_recovery"] = config["context"].get("recovery", {})
        sentinel_config = config["context"].get("sentinel", {})
        summary["belief_sentinel_enabled"] = sentinel_config.get("enabled", False)
        consistency_enabled = sentinel_lane_enabled(sentinel_config, "consistency")
        progress_enabled = sentinel_lane_enabled(sentinel_config, "progress")
        summary["belief_sentinel_consistency_enabled"] = consistency_enabled
        summary["belief_sentinel_progress_enabled"] = progress_enabled
        if consistency_enabled:
            consistency_model = sentinel_model_config(sentinel_config, "consistency")
            summary["belief_sentinel_consistency_model"] = consistency_model["model"]
            summary["belief_sentinel_local_audit_every_steps"] = sentinel_config.get(
                "local_audit_every_steps", 1
            )
            summary["belief_sentinel_global_audit_every_steps"] = sentinel_config.get(
                "global_audit_every_steps", 8
            )
        if (
            progress_enabled
            and config["context"].get("task_mode", "execution") == "execution"
        ):
            progress_model = sentinel_model_config(sentinel_config, "progress")
            summary["belief_sentinel_progress_model"] = progress_model["model"]
    return summary


def write_run_overview(
    output: Path,
    config: dict,
    selected_context: str,
    compressor_config: dict | None,
    results: list[dict],
    case_times: list[float],
    case_paths: list[Path],
) -> None:
    """Rebuild the shared run summary under an interprocess file lock."""
    with (output / ".overview.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        persisted_results = []
        persisted_times = []
        persisted_paths = []
        for persisted_path in sorted((output / "cases").glob("case_*")):
            result_path = persisted_path / "result.json"
            if not result_path.is_file():
                continue
            try:
                persisted_result = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            persisted_results.append(persisted_result)
            persisted_times.append(
                completed_case_seconds(persisted_result, persisted_path)
            )
            persisted_paths.append(persisted_path)
        if persisted_results:
            results = persisted_results
            case_times = persisted_times
            case_paths = persisted_paths
        summary = build_summary(
            config, selected_context, compressor_config, results, case_times
        )
        summary_path = output / "summary.json"
        temporary_summary = output / f".summary.{os.getpid()}.tmp"
        temporary_summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_summary.replace(summary_path)
        render_case_index_markdown(case_paths, output / "events.md")


def context_type(config: dict) -> str:
    """Read the context type; configurations without a context use Raw trajectory."""
    if "context" in config:
        return config["context"]["type"].lower()
    return "raw"


def create_context(
    config: dict,
    call_compressor,
    call_belief,
    call_sentinel_consistency,
    logger: Logger,
    call_sentinel_progress=None,
):
    """Create the Raw trajectory or PoS belief context selected by the configuration."""
    selected = context_type(config)
    if selected == "raw":
        return RawTrajectoryContext()
    if selected == "belief":
        return BeliefManager(
            config,
            call_belief,
            logger,
            call_sentinel=call_sentinel_consistency,
            call_progress=call_sentinel_progress,
        )
    raise ValueError(f"Unsupported context: {selected}")


def create_method(
    config: dict,
    call_model,
    call_compressor,
    logger: Logger,
    call_belief=None,
    call_sentinel_consistency=None,
    call_sentinel_progress=None,
):
    """Compose the ReAct policy loop with the selected context provider."""
    harness = config.get("harness", "ReAct")
    context_provider = create_context(
        config,
        call_compressor,
        call_belief,
        call_sentinel_consistency,
        logger,
        call_sentinel_progress,
    )
    if harness == "ReAct":
        return ReAct(config, call_model, logger, context_provider=context_provider)
    raise ValueError(f"Unsupported harness: {harness}")


def completed_case_count(output: Path, total: int, start_case: int = 0) -> int:
    """Count complete result files in the requested half-open case range."""
    cases_output = output / "cases"
    return sum(
        (
            (cases_output / f"case_{index:04d}" / "result.json").is_file()
            for index in range(start_case, start_case + total)
        )
    )


def supervise_experiment(config_path: str, config: dict) -> None:
    """Restart a failed worker until every case in the selected range has a result."""
    if not config.get("skip_existing", False):
        raise ValueError("run_until_complete requires skip_existing: true")
    output = output_directory(config)
    start_case = int(config["start_case"])
    total = int(config["end_case"]) - start_case
    retry_delay = config.get("restart_delay_seconds", 10)
    while completed_case_count(output, total, start_case) < total:
        completed = completed_case_count(output, total, start_case)
        print(f"Completed {completed}/{total} cases; starting worker.", flush=True)
        worker_environment = {**os.environ, WORKER_ENV: "1"}
        worker = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--config", config_path],
            env=worker_environment,
        )
        completed = completed_case_count(output, total, start_case)
        print(
            f"Worker exited with code {worker.returncode}; completed {completed}/{total} cases.",
            flush=True,
        )
        if completed < total:
            print(f"Restarting worker in {retry_delay} seconds.", flush=True)
            sleep(retry_delay)
    print(f"All {total} cases completed.", flush=True)


def run_experiment(config: dict) -> None:
    """Run the selected cases, preserving completed results when resume is enabled."""
    output = output_directory(config)
    output.mkdir(parents=True, exist_ok=True)
    cases_output = output / "cases"
    cases_output.mkdir(parents=True, exist_ok=True)
    adapter = Adapter(config)
    call_model = create_model(config)
    selected_context = context_type(config)
    compressor_config = (
        config["compressor"]
        if "compressor" in config
        else config.get("context", {}).get("model")
    )
    call_compressor = None
    call_belief = (
        create_model(config["context"]["model"])
        if selected_context == "belief"
        else None
    )
    sentinel_config = config.get("context", {}).get("sentinel", {})
    call_sentinel_consistency = (
        create_model(sentinel_model_config(sentinel_config, "consistency"))
        if selected_context == "belief"
        and sentinel_lane_enabled(sentinel_config, "consistency")
        else None
    )
    task_mode = config.get("context", {}).get("task_mode", "execution")
    call_sentinel_progress = (
        create_model(sentinel_model_config(sentinel_config, "progress"))
        if selected_context == "belief"
        and sentinel_lane_enabled(sentinel_config, "progress")
        and (task_mode == "execution")
        else None
    )
    results = []
    case_times = []
    case_paths = []
    skip_existing = config.get("skip_existing", False)
    start_case = int(config.get("start_case", 0))
    try:
        progress = tqdm(
            range(len(adapter)),
            desc=f"{config['benchmark']}/{config['method']}",
            unit="case",
        )
        for local_index in progress:
            index = start_case + local_index
            case_path = cases_output / f"case_{index:04d}"
            result_path = case_path / "result.json"
            if skip_existing and result_path.exists():
                result = load_completed_case(adapter, case_path)
                markdown_path = case_path / "events.md"
                events_path = case_path / "events.jsonl"
                if not markdown_path.exists() and events_path.exists():
                    render_events_markdown(events_path, markdown_path)
                results.append(result)
                case_times.append(completed_case_seconds(result, case_path))
                case_paths.append(case_path)
                write_run_overview(
                    output,
                    config,
                    selected_context,
                    compressor_config,
                    results,
                    case_times,
                    case_paths,
                )
                progress.set_postfix(
                    success=result["success"], steps=result["steps"], skipped=True
                )
                continue
            case_path.mkdir(parents=True, exist_ok=True)
            logger = Logger(case_path / "events.jsonl")
            adapter.attach_logger(logger)
            try:
                method = create_method(
                    config,
                    call_model,
                    call_compressor,
                    logger,
                    call_belief,
                    call_sentinel_consistency,
                    call_sentinel_progress,
                )
                token_usage_before = collect_token_usage(
                    call_model,
                    call_compressor,
                    call_belief,
                    call_sentinel_consistency,
                    call_sentinel_progress,
                )
                start = perf_counter()
                result = method.run(adapter)
            finally:
                logger.close()
            case_seconds = perf_counter() - start
            case_times.append(case_seconds)
            result["case_seconds"] = round(case_seconds, 3)
            result["token_usage"] = subtract_token_usage(
                collect_token_usage(
                    call_model,
                    call_compressor,
                    call_belief,
                    call_sentinel_consistency,
                    call_sentinel_progress,
                ),
                token_usage_before,
            )
            results.append(result)
            case_paths.append(case_path)
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            render_events_markdown(case_path / "events.jsonl", case_path / "events.md")
            write_run_overview(
                output,
                config,
                selected_context,
                compressor_config,
                results,
                case_times,
                case_paths,
            )
            adapter.finalize_case(bool(result["success"]))
            progress.set_postfix(success=result["success"], steps=result["steps"])
    finally:
        adapter.close()


def main() -> None:
    """Main."""
    parser = argparse.ArgumentParser(description="Run a PoS baseline experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    args = parser.parse_args()
    config = load_config(args.config)
    is_worker = os.getenv(WORKER_ENV) == "1"
    if config.get("run_until_complete", False) and (not is_worker):
        supervise_experiment(args.config, config)
        return
    run_experiment(config)


if __name__ == "__main__":
    main()
