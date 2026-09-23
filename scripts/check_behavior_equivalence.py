"""Compare deterministic Raw/PoS behavior against a separately supplied source tree.

Both trees run in isolated subprocesses with fixed model responses and a mock
environment. No benchmark data, model API, or runtime output directory is used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def snapshot(root: Path) -> dict:
    """Record prompts, actions, committed beliefs, health, recovery, and results."""
    sys.path.insert(0, str(root.resolve()))
    from baselines.react import ReAct
    from belief.manager import BeliefManager
    from contexts.raw import RawTrajectoryContext

    class Recorder:
        def __init__(self):
            self.events = []

        def log(self, event, **data):
            self.events.append({"event": event, **data})

    class Environment:
        def __init__(self):
            self.actions = []

        def reset(self):
            return self.state()

        def state(self):
            return {
                "case_id": "fixed/case",
                "task": "Clean mug 1.",
                "observation": "Mug 1 remains dirty.",
                "actions": ["inspect", "clean"],
                "done": len(self.actions) >= 3,
                "success": False,
                "score": 0.0,
            }

        def step(self, action):
            self.actions.append(action)
            return self.state()

    def run(context_kind, consistency=True, trapping=True, task_mode="execution"):
        calls = []
        initial = {
            "goal_specification": "Mug 1 must be clean.",
            "entities": [{"entity_id": "e1", "entity_type": "mug", "name": "mug 1"}],
            "states": [
                {
                    "state_id": "s1",
                    "entity_id": "e1",
                    "description": "Mug 1 is dirty.",
                    "source_type": "observed",
                    "probability": 1.0,
                }
            ],
            "relations": [],
            "epistemic_gaps": [],
            "achievement_gaps": [
                {"target": "Mug 1 must be clean", "reason": "Mug 1 is dirty."}
            ],
            "frontier": {"gap_type": "achievement", "target": "Mug 1 must be clean"},
        }
        delta = {
            "remove_state_ids": [],
            "remove_relation_ids": [],
            "upsert_entities": [],
            "upsert_states": [],
            "upsert_relations": [],
            "epistemic_gaps": [],
            "achievement_gaps": initial["achievement_gaps"],
            "frontier": initial["frontier"],
        }

        def caller(role):
            def call(messages):
                calls.append({"role": role, "messages": messages})
                if role == "agent":
                    return json.dumps(
                        {
                            "thought": "Inspect then clean.",
                            "actions": ["inspect", "clean"],
                        }
                    )
                if role == "consistency":
                    return '{"issues": []}'
                if role == "progress":
                    return '{"progress": 0, "score": 0.0, "affected_record_ids": [], "reason": "No new evidence."}'
                if "coherent current-world description" in messages[0]["content"]:
                    return '{"belief_text": "Mug 1 remains dirty and must be cleaned."}'
                return json.dumps(
                    initial
                    if "Initial observation:" in messages[-1]["content"]
                    else delta
                )

            return call

        config = {
            "benchmark": "ALFWorld",
            "method": "ReAct-PoS" if context_kind == "belief" else "ReAct",
            "harness": "ReAct",
            "max_steps": 3,
            "max_actions_per_turn": 2,
            "max_json_retries": 0,
            "include_actions": True,
            "context": {
                "type": context_kind,
                "task_mode": task_mode,
                "agent_view": "hybrid",
                "update_every_steps": 1,
                "model": {"max_json_retries": 0},
                "sentinel": {
                    "enabled": True,
                    "consistency_enabled": consistency,
                    "progress_enabled": True,
                    "global_audit_every_steps": 2,
                    "model": {"max_json_retries": 0},
                },
                "trapping": {
                    "enabled": trapping,
                    "window_size": 2,
                    "health_threshold": 0.25,
                    "cycle_threshold": 0.75,
                    "recurrence_epsilon": 0.15,
                    "max_cycle_length": 1,
                },
                "recovery": {"enabled": trapping},
            },
        }
        logger = Recorder()
        context = (
            BeliefManager(
                config,
                caller("belief"),
                logger,
                call_sentinel=caller("consistency"),
                call_progress=caller("progress"),
            )
            if context_kind == "belief"
            else RawTrajectoryContext()
        )
        environment = Environment()
        result = ReAct(config, caller("agent"), logger, context_provider=context).run(
            environment
        )
        assert len(environment.actions) == 3, (
            "Fixture must exercise real action transitions"
        )
        if context_kind == "belief" and trapping:
            assert any("recovery" in item["event"] for item in logger.events), (
                "Fixture must exercise recovery"
            )
        return {
            "calls": calls,
            "actions": environment.actions,
            "result": result,
            "events": logger.events,
        }

    return {
        "raw": run("raw"),
        "pos": run("belief"),
        "no_consistency": run("belief", consistency=False),
        "no_trapping": run("belief", trapping=False),
        "diagnostic": run("belief", task_mode="diagnostic"),
    }


def main():
    """Compare source trees without placing test artifacts in either tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the five offline fixed-response scenarios",
    )
    parser.add_argument(
        "--candidate", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        snapshot(args.candidate)
        print("Passed: Raw, full PoS, both ablations, and diagnostic PoS.")
        return
    if args.snapshot:
        print(json.dumps(snapshot(args.snapshot), sort_keys=True))
        return
    if args.reference is None:
        parser.error("--reference is required")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    outputs = [
        subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), "--snapshot", str(root)],
            env=env,
            text=True,
        )
        for root in (args.reference, args.candidate)
    ]
    if json.loads(outputs[0]) != json.loads(outputs[1]):
        raise SystemExit(
            "Behavior differs: inspect snapshots with --snapshot before release."
        )
    print(
        "Equivalent: Raw, full PoS, both ablations, and diagnostic PoS (five fixed-response scenarios)."
    )


if __name__ == "__main__":
    main()
