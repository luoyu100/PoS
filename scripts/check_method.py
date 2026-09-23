"""Run offline checks of belief health, progress, and recovery semantics."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
from dataclasses import asdict
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from belief.beliefStates import (
    AchievementGap,
    BeliefHealth,
    BeliefState,
    EpistemicGap,
    GapFrontier,
    Goal,
    HealthTransition,
)
from belief.manager import BeliefManager
from belief.progressScoring import diagnostic_progress
from belief.recoveryPlanning import compose_recovery_instruction
from belief.trappingDetection import StableGapIdentityTracker, detect_belief_trapping
from belief.worldStates import WorldState


def world(value):
    return {
        "entities": {"e1": {"entity_id": "e1", "name": str(value)}},
        "states": {
            "s1": {
                "state_id": "s1",
                "entity_id": "e1",
                "description": str(value),
                "probability": 0.5,
            }
        },
        "relations": {
            "r1": {
                "relation_id": "r1",
                "source_id": "e1",
                "target_id": "s1",
                "description": str(value),
            }
        },
    }


def detect(worlds, progress, e=None, a=None, active_gap=None, **settings):
    n = len(worlds)
    snapshots = [
        {
            "step": i,
            "world": w,
            "epistemic_gap_ids": e[i] if e is not None else ["e:1"],
            "achievement_gap_ids": a[i] if a is not None else [],
        }
        for i, w in enumerate(worlds)
    ]
    kwargs = dict(
        window_size=n,
        max_cycle_length=min(2, n - 1),
        health_threshold=0.25,
        cycle_threshold=0.75,
        recurrence_epsilon=0.15,
    )
    kwargs.update(settings)
    return detect_belief_trapping(
        [{"step": i, "progress": p} for i, p in enumerate(progress)],
        snapshots,
        active_gap=active_gap,
        **kwargs,
    )


class HealthTests(unittest.TestCase):
    def test_clindiag_example_is_format_only(self):
        from prompts.react_prompt import CLINDIAG_EXAMPLE, react_example

        manifest = (
            (Path(__file__).resolve().parents[1] / "benchmarks/ClinDiag/manifest.txt")
            .read_text()
            .splitlines()
        )
        self.assertIn("Fictional format-only", CLINDIAG_EXAMPLE)
        self.assertNotIn("case_id", CLINDIAG_EXAMPLE)
        for case_id in manifest:
            if case_id.strip():
                self.assertNotIn(case_id.strip(), CLINDIAG_EXAMPLE)
        self.assertIn(
            '"final_diagnosis": "<specific evidence-supported diagnosis>"',
            CLINDIAG_EXAMPLE,
        )
        self.assertEqual(react_example({"benchmark": "ClinDiag"}), CLINDIAG_EXAMPLE)

    def test_rca_raw_has_no_active_gap_instruction(self):
        from prompts.react_prompt import build_prompt

        messages = build_prompt(
            {"task": "Investigate an incident", "observation": "Ready", "actions": []},
            [],
            {"benchmark": "RCA100", "include_actions": True, "max_actions_per_turn": 3},
        )
        self.assertNotIn(
            "active gap", "\n".join(m["content"] for m in messages).lower()
        )

    def test_loca_shared_tool_catalog(self):
        import json
        from prompts.react_prompt import (
            build_prompt,
            build_belief_prompt,
            build_loca_tool_catalog,
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "files_read",
                    "description": (
                        "[filesystem] Read a file.\n"
                        + "Keep the full tool instructions. " * 30
                        + "\nArgs: path must be absolute. Returns: text."
                    ),
                    "strict": True,
                    "parameters": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Absolute path to the input file.",
                                "pattern": "^/",
                                "minLength": 1,
                                "maxLength": 4096,
                            },
                            "options": {"$ref": "#/$defs/Options"},
                        },
                        "$defs": {
                            "Options": {
                                "type": "object",
                                "properties": {"encoding": {"const": "utf-8"}},
                                "additionalProperties": False,
                            }
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        state = {"task": "Inspect a file", "observation": "Ready", "actions": tools}
        context = {
            "belief_text": "Ready",
            "goal": {},
            "world": {},
            "frontier_gap": None,
        }
        original_tools = copy.deepcopy(tools)
        for enabled in (True, False):
            config = {"benchmark": "LOCA-Bench", "include_actions": enabled}
            raw = build_prompt(state, [], config)[-1]["content"]
            pos = build_belief_prompt(state, context, config)[-1]["content"]
            raw_catalog = json.loads(
                raw.split("\nAvailable actions:\n", 1)[1].split(
                    "\n\nPrevious steps:", 1
                )[0]
            )
            pos_catalog = json.loads(
                pos.split("\nAvailable action schemas:\n", 1)[1].split("\n\nChoose", 1)[
                    0
                ]
            )
            self.assertEqual(raw_catalog, pos_catalog)
            self.assertEqual(
                raw_catalog, build_loca_tool_catalog(tools if enabled else [])
            )
            entries = [
                tool
                for group in raw_catalog["tools_by_role"].values()
                for tool in group
            ]
            self.assertEqual(entries, [tools[0]["function"]] if enabled else [])
        self.assertEqual(tools, original_tools)

    def test_boundary_configurations(self):
        for worlds, progress, expected in (
            ([world(i) for i in range(4)], [0] * 4, 0),
            ([world(0)] * 4, [1] * 4, 0),
            ([world(i) for i in range(4)], [1] * 4, 1),
        ):
            with self.subTest(expected=expected, progress=progress):
                self.assertEqual(detect(worlds, progress).health_score, expected)

    def test_initial_gap_denominator(self):
        result = detect([world(0)] * 4, [0] * 4, e=[["x", "y"], ["x"], ["x"], ["x"]])
        self.assertEqual(result.persistence_e, 0.5)
        self.assertEqual(result.health_score, 0.5)

    def test_initial_empty(self):
        result = detect([world(0)] * 4, [0] * 4, e=[[], ["x"], ["x"], ["x"]])
        self.assertEqual(result.persistence_e, 0)

    def test_recurrence_excludes_gaps(self):
        result = detect([world(0)] * 4, [0] * 4, e=[[str(i)] for i in range(4)])
        self.assertEqual(result.recurrence, 1)

    def test_static_precedence(self):
        self.assertEqual(detect([world(0)] * 4, [0] * 4).trapping_pattern, "static")

    def test_cycle(self):
        result = detect([world(0), world(1)] * 2, [0] * 4)
        self.assertEqual(
            (result.recurrence_period, result.trapping_pattern), (2, "cycle")
        )

    def test_cycle_threshold(self):
        result = detect([world(0), world(1), world(0), world(2)], [0] * 4)
        self.assertEqual(result.recurrence, 0.5)
        self.assertIsNone(result.trapping_pattern)

    def test_drift(self):
        worlds = []
        for i in range(4):
            w = world("focus")
            w["entities"]["e2"] = {"entity_id": "e2", "name": f"other{i}"}
            worlds.append(w)
        result = detect(worlds, [0] * 4, active_gap={"target": "focus", "reason": ""})
        self.assertEqual(result.trapping_pattern, "drift")

    def test_generic_recovery(self):
        result = detect([world(i) for i in range(4)], [0] * 4)
        self.assertIsNone(result.trapping_pattern)
        self.assertEqual(result.health_score, 0)
        text = compose_recovery_instruction(
            pattern=None,
            gap_type="epistemic",
            gap={"target": "check signal"},
            recent_transitions=[],
            recurrence_period=None,
        )
        self.assertIn("Reassess", text)

    def test_tie_epistemic(self):
        result = detect(
            [world(0)] * 4,
            [0] * 4,
            a=[["a:1"]] * 4,
            active_gap={"target": "0", "gap_type": "achievement"},
        )
        self.assertEqual(result.gap_dimension, "epistemic")

    def test_confidence_affects_recurrence(self):
        worlds = [world(0) for _ in range(4)]
        for i, w in enumerate(worlds):
            w["states"]["s1"]["probability"] = i / 4
        self.assertEqual(detect(worlds, [0] * 4).recurrence, 0)

    def test_field_order_is_irrelevant(self):
        w = world(0)
        shuffled = {
            k: {i: dict(reversed(list(v.items()))) for i, v in records.items()}
            for k, records in w.items()
        }
        self.assertEqual(detect([w, shuffled], [0, 0]).recurrence, 1)

    def test_unverified_not_zero(self):
        self.assertFalse(detect([world(0)] * 4, [0, None, 0, 0]).window_ready)

    def test_invalid_lag(self):
        with self.assertRaises(ValueError):
            detect([world(0)] * 4, [0] * 4, max_cycle_length=4)

    def test_independent_confidence(self):
        old = {
            "world": {"states": {"x": {"source_type": "inferred", "probability": 0.2}}}
        }
        new = copy.deepcopy(old)
        new["world"]["states"]["x"]["probability"] = 0.95
        result = diagnostic_progress(old, new, None, 0.3)
        self.assertEqual((result["score"], result["progress"]), (0.375, 1))
        old["world"]["relations"] = {
            "x": {"source_type": "inferred", "probability": 0.1}
        }
        new["world"]["relations"] = {
            "x": {"source_type": "inferred", "probability": 0.9}
        }
        self.assertEqual(diagnostic_progress(old, new, None, 0.3)["score"], 0.775)

    def test_recovery_refresh_and_release(self):
        manager = BeliefManager.__new__(BeliefManager)
        manager.belief_state = BeliefState(
            WorldState(),
            Goal("goal", "goal"),
            epistemic_gaps=[EpistemicGap("signal", "unknown")],
            frontier=GapFrontier("epistemic", "signal"),
            health=BeliefHealth(
                window_ready=True,
                health_score=0,
                gap_dimension="epistemic",
                trapping_pattern="static",
            ),
        )
        manager.recovery_config = {"enabled": True}
        manager.trapping_config = {"window_size": 4}
        manager.active_recovery = None
        manager.recovery_history = []
        manager.health_transitions = []
        manager.case_id = "mock"
        manager.logger = type("Logger", (), {"log": lambda *args, **kwargs: None})()
        manager._update_recovery()
        original = manager.active_recovery
        manager.belief_state.health.trapping_pattern = None
        manager._update_recovery()
        self.assertIsNot(manager.active_recovery, original)
        self.assertEqual(
            manager.active_recovery.recovery_frontier, manager.belief_state.frontier
        )
        self.assertIn("Reassess", manager.active_recovery.instruction)
        manager.belief_state.health.health_score = 0.26
        manager._update_recovery()
        self.assertIsNone(manager.active_recovery)

    def test_terminal_has_no_next_recovery(self):
        manager = BeliefManager.__new__(BeliefManager)
        manager.task_done = True
        manager.case_id = "mock"
        manager.active_recovery = None
        manager.health_transitions = [HealthTransition(0, 0, None, "finish", "done")]
        manager.logger = type("Logger", (), {"log": lambda *args, **kwargs: None})()
        manager._update_health = lambda: self.fail(
            "Terminal transition must not trigger recovery"
        )
        finding = SimpleNamespace(
            step=0,
            progress=1,
            scoring_method="mock",
            score=1,
            affected_record_ids=(),
            reason="done",
            action="finish",
        )
        manager._apply_progress_finding(finding)
        self.assertEqual(manager.health_transitions[0].progress, 1)

    def test_recovery_targets_blocked_dimension(self):
        for active_type in ("epistemic", "achievement"):
            for blocked_type in ("epistemic", "achievement"):
                with self.subTest(active=active_type, blocked=blocked_type):
                    manager = self._recovery_manager(active_type, blocked_type)
                    active = copy.deepcopy(manager.belief_state.frontier)
                    health = copy.deepcopy(manager.belief_state.health)
                    manager._start_recovery()
                    recovery = manager.active_recovery
                    self.assertEqual(recovery.recovery_frontier.gap_type, blocked_type)
                    self.assertEqual(recovery.gap_dimension, blocked_type)
                    self.assertEqual(recovery.trigger_frontier, active)
                    self.assertEqual(manager.belief_state.frontier, active)
                    self.assertEqual(manager.belief_state.health, health)
                    self.assertEqual(
                        manager._recovery_context()["recovery_gap"]["gap_type"],
                        blocked_type,
                    )
                    self.assertIn(f"{blocked_type} recovery gap", recovery.instruction)

    def test_recovery_prefers_persistent_gap(self):
        manager = self._recovery_manager("epistemic", "achievement")
        manager.belief_state.achievement_gaps = [
            AchievementGap("Return book to shelf", "Book is on desk"),
            AchievementGap("Wash dirty mug", "Mug is dirty"),
        ]
        tracked = manager._gap_identity_tracker.reconcile(
            "achievement", manager.belief_state.achievement_gaps
        )
        manager.belief_state.health.persistent_achievement_gaps = (tracked[1].identity,)
        manager._start_recovery()
        self.assertEqual(
            manager.active_recovery.recovery_frontier.target, "Wash dirty mug"
        )
        self.assertEqual(manager.belief_state.frontier.gap_type, "epistemic")

    def test_recovery_does_not_fall_back_to_wrong_dimension(self):
        manager = self._recovery_manager("epistemic", "achievement")
        manager.belief_state.achievement_gaps = []
        manager._start_recovery()
        self.assertIsNone(manager.active_recovery)

    @staticmethod
    def _recovery_manager(active_type, blocked_type):
        manager = BeliefManager.__new__(BeliefManager)
        targets = {
            "epistemic": "Is the signal present?",
            "achievement": "Wash dirty mug",
        }
        manager.belief_state = BeliefState(
            WorldState(),
            Goal("goal", "goal"),
            epistemic_gaps=[EpistemicGap(targets["epistemic"], "Unknown signal")],
            achievement_gaps=[AchievementGap(targets["achievement"], "Mug is dirty")],
            frontier=GapFrontier(active_type, targets[active_type]),
            health=BeliefHealth(
                window_ready=True,
                health_score=0,
                gap_dimension=blocked_type,
                trapping_pattern="static",
            ),
        )
        manager.recovery_config = {"enabled": True}
        manager.trapping_config = {"window_size": 4}
        manager.active_recovery = None
        manager.health_transitions = []
        manager._gap_identity_tracker = StableGapIdentityTracker()
        manager.case_id = "mock"
        manager.logger = SimpleNamespace(log=lambda *args, **kwargs: None)
        return manager

    def test_terminal_context_errors_preserve_outcome_for_all_methods(self):
        from baselines.react import ReAct
        from utils.public_function import _render_event_steps

        for method in ("raw", "pos", "no_consistency", "no_trapping"):
            for success in (False, True):
                for failing_phases in (
                    (),
                    ("update",),
                    ("export",),
                    ("update", "export"),
                ):
                    with self.subTest(
                        method=method, success=success, failures=failing_phases
                    ):
                        config, environment, context, logger = self._terminal_fixture(
                            method, success, failing_phases
                        )
                        agent = ReAct(
                            config,
                            lambda _: (
                                '{"thought":"Submit once.","actions":["finish","unexpected"]}'
                            ),
                            logger,
                            context,
                        )
                        result = agent.run(environment)
                        self.assertEqual(environment.calls, 1)
                        self.assertEqual(context.updates, 1)
                        self.assertEqual(result["success"], success)
                        self.assertEqual(result["score"], 1.0 if success else 0.25)
                        self.assertEqual(result["steps"], 1)
                        self.assertEqual(result["trajectory"][0]["action"], "finish")
                        self.assertEqual(
                            [
                                error["phase"]
                                for error in result.get("context_errors", [])
                            ],
                            list(failing_phases),
                        )
                        error_events = [
                            e
                            for e in logger.events
                            if e["event"] == "terminal_context_error"
                        ]
                        self.assertEqual(len(error_events), len(failing_phases))
                        if failing_phases:
                            self.assertIn(
                                "Terminal context error",
                                "\n".join(_render_event_steps(error_events, 3)),
                            )

    def test_nonterminal_context_failure_still_propagates(self):
        from baselines.react import ReAct

        for method in ("raw", "pos", "no_consistency", "no_trapping"):
            config, environment, context, logger = self._terminal_fixture(
                method, False, ("update",), terminal=False
            )
            agent = ReAct(
                config,
                lambda _: '{"thought":"Inspect.","action":"inspect"}',
                logger,
                context,
            )
            with self.assertRaisesRegex(RuntimeError, "update failure"):
                agent.run(environment)
            self.assertEqual(environment.calls, 1)
            self.assertFalse(
                any(e["event"] == "terminal_context_error" for e in logger.events)
            )

    def test_terminal_belief_model_failure_preserves_result(self):
        from baselines.react import ReAct

        initial = {
            "goal_specification": "Complete the task",
            "entities": [],
            "states": [],
            "relations": [],
            "epistemic_gaps": [
                {"target": "Is the signal present?", "reason": "Not observed"}
            ],
            "achievement_gaps": [],
            "frontier": {
                "gap_type": "epistemic",
                "target": "Is the signal present?",
            },
        }
        for method in ("pos", "no_consistency", "no_trapping"):
            for success in (False, True):
                with self.subTest(method=method, success=success):
                    config, environment, _, logger = self._terminal_fixture(
                        method, success, ()
                    )
                    responses = iter(
                        (json.dumps(initial), '{"belief_text":"Signal unknown."}')
                    )

                    def belief_model(messages):
                        response = next(responses, None)
                        if response is None:
                            raise RuntimeError("Terminal belief model unavailable")
                        return response

                    manager = BeliefManager(
                        config,
                        belief_model,
                        logger,
                        call_sentinel=lambda _: '{"issues":[]}',
                        call_progress=lambda _: '{"progress":0,"reason":"None"}',
                    )
                    result = ReAct(
                        config,
                        lambda _: '{"thought":"Submit.","action":"finish"}',
                        logger,
                        manager,
                    ).run(environment)
                    self.assertEqual(result["success"], success)
                    self.assertEqual(result["score"], 1.0 if success else 0.25)
                    self.assertEqual(result["steps"], 1)
                    self.assertEqual(result["raw_trajectory"][-1]["action"], "finish")
                    self.assertEqual(result["context_errors"][0]["phase"], "update")
                    self.assertEqual(result["health_transitions"], [])

    @staticmethod
    def _terminal_fixture(method, success, failing_phases, terminal=True):
        class Environment:
            calls = 0

            def reset(self):
                return {
                    "case_id": "mock",
                    "task": "Complete the task",
                    "observation": "Ready",
                    "actions": ["finish"],
                    "done": False,
                    "success": False,
                    "score": 0.0,
                }

            def step(self, action):
                self.calls += 1
                return {
                    **self.reset(),
                    "done": terminal,
                    "success": success,
                    "score": 1.0 if success else 0.25,
                }

        class Context:
            updates = 0

            def reset(self, state):
                pass

            def get_context(self):
                if method == "raw":
                    return {"type": "raw", "trajectory": []}
                return {
                    "type": "belief",
                    "belief_text": "Ready",
                    "goal": {},
                    "world": {},
                    "frontier_gap": None,
                }

            def update(self, transition, state, step):
                self.updates += 1
                if "update" in failing_phases:
                    raise RuntimeError("update failure")

            def get_result(self):
                if "export" in failing_phases:
                    raise RuntimeError("export failure")
                return {}

            def close(self):
                pass

        events = []
        logger = SimpleNamespace(
            events=events,
            log=lambda event, **data: events.append({"event": event, **data}),
        )
        config = {
            "benchmark": "RCA100",
            "max_steps": 2,
            "max_actions_per_turn": 2,
            "max_json_retries": 0,
            "include_actions": True,
            "context": {
                "type": "raw" if method == "raw" else "belief",
                "sentinel": {
                    "enabled": True,
                    "consistency_enabled": method != "no_consistency",
                },
                "trapping": {"enabled": method != "no_trapping"},
            },
        }
        return config, Environment(), Context(), logger

    def test_cycle_evidence(self):
        manager = BeliefManager.__new__(BeliefManager)
        manager.belief_state = BeliefState(
            WorldState(),
            Goal("goal", "goal"),
            health=BeliefHealth(trapping_pattern="cycle", recurrence_period=2),
        )
        manager.trapping_config = {"window_size": 4, "recurrence_epsilon": 0.15}
        manager.health_transitions = [
            HealthTransition(i, 0, None, f"action{i}", "obs", progress=1)
            for i in range(4)
        ]
        manager.belief_snapshots = [
            {"step": i, "world": world(i % 2)} for i in range(-1, 4)
        ]
        manager._frontier_context = lambda: None
        evidence = manager._cycle_transitions()
        self.assertEqual([item["action"] for item in evidence], ["action2", "action3"])
        self.assertEqual(evidence[0]["world_before"], world(1))

    def test_health_markdown(self):
        from utils.public_function import _render_event_steps

        text = "\n".join(
            _render_event_steps(
                [{"event": "belief_health_updated", "health": asdict(BeliefHealth())}],
                level=3,
            )
        )
        self.assertIn("H: 1.000", text)
        self.assertNotIn("BTS", text)


if __name__ == "__main__":
    unittest.main()
