"""Benchmark-independent ReAct policy loop with pluggable Raw or PoS context."""

from __future__ import annotations
import json
from collections.abc import Callable
from contexts.base import ContextProvider
from contexts.raw import RawTrajectoryContext
from prompts.react_prompt import (
    append_final_turn_directive,
    build_belief_prompt,
    build_prompt,
    build_repair_prompt,
)
from utils.logger import Logger

MISSING_ACTION_DIRECTIVE = 'Your previous JSON did not contain an executable environment action.\nThe environment is still running, so an "answer" field or a natural-language completion claim cannot finish the case.\nReturn one new decision that follows the system prompt\'s exact action schema and selects an action verbatim from Available actions.\nUse operation "act" when the current context protocol requires an operation field.\nTreat object and entity names as exact: a similarly named object does not satisfy the requested target.\nDo not return an answer, final response, or another completion claim without an executable action.\n'


class ReAct:
    """Execute thought-action-observation decisions until termination or the turn budget."""

    def __init__(
        self,
        config: dict,
        call_model: Callable[[list[dict]], str],
        logger: Logger,
        context_provider: ContextProvider | None = None,
    ):
        """Initialize configuration and episode-local runtime state."""
        self.config = config
        self.call_model = call_model
        self.logger = logger
        self.context_provider = context_provider or RawTrajectoryContext()

    def run(self, adapter) -> dict:
        """Run one episode and return the full interaction trajectory and benchmark outcome."""
        state = adapter.reset()
        trajectory = []
        context_errors = []
        try:
            self.context_provider.reset(state)
            step = 0
            turn = 0
            note_turn = getattr(adapter, "note_turn", None)
            while turn < self.config["max_steps"]:
                if note_turn is not None:
                    note_turn(turn, self.config["max_steps"])
                messages = self._build_messages(state, turn=turn)
                if turn == self.config["max_steps"] - 1:
                    messages = append_final_turn_directive(messages, self.config)
                decision = self._request_decision(
                    messages, case_id=state["case_id"], step=step
                )
                self.logger.log(
                    "react_model_decision",
                    case_id=state["case_id"],
                    step=step,
                    decision_source="model",
                    decision=decision,
                )
                if not self._has_executable_action(decision):
                    decision = self._decide_after_rejected_operation(
                        messages=messages,
                        decision=decision,
                        state=state,
                        step=step,
                        event="react_missing_action",
                        directive=MISSING_ACTION_DIRECTIVE,
                    )
                    if decision is None:
                        turn += 1
                        continue
                for action in self._decision_actions(decision):
                    state = adapter.step(action)
                    transition = {
                        "thought": decision["thought"],
                        "action": action,
                        "observation": state["observation"],
                    }
                    trajectory.append(transition)
                    try:
                        self.context_provider.update(transition, state, step)
                    except Exception as error:
                        if not state["done"]:
                            raise
                        context_errors.append(
                            self._record_terminal_context_error(
                                state, step, "update", error
                            )
                        )
                    step += 1
                    if state["done"]:
                        break
                turn += 1
                if state["done"]:
                    break
            result = {
                "case_id": state["case_id"],
                "task": state["task"],
                "success": state["success"],
                "score": state["score"],
                "steps": len(trajectory),
                "final_observation": state["observation"],
                "trajectory": trajectory,
            }
            if "benchmark_info" in state:
                result["benchmark_info"] = state["benchmark_info"]
            try:
                result.update(self.context_provider.get_result())
            except Exception as error:
                if not state["done"]:
                    raise
                context_errors.append(
                    self._record_terminal_context_error(
                        state, step - 1, "export", error
                    )
                )
            if context_errors:
                result["context_errors"] = context_errors
            return result
        except Exception:
            close_context = getattr(self.context_provider, "close", None)
            if close_context is not None:
                close_context()
            raise

    def _record_terminal_context_error(
        self, state: dict, step: int, phase: str, error: Exception
    ) -> dict:
        """Record a context failure without changing an authoritative terminal result."""
        detail = {
            "step": step,
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        self.logger.log(
            "terminal_context_error",
            case_id=state["case_id"],
            **detail,
        )
        return detail

    def _decide_after_rejected_operation(
        self,
        messages: list[dict],
        decision: dict,
        state: dict,
        step: int,
        event: str,
        directive: str,
        **log_fields,
    ) -> dict | None:
        """Request one executable replacement decision; charge a turn if still invalid."""
        self.logger.log(
            event,
            case_id=state["case_id"],
            step=step,
            rejected_decision=decision,
            **log_fields,
        )
        retry_messages = [
            *messages,
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
            {"role": "user", "content": directive},
        ]
        retried = self._request_decision(
            retry_messages, case_id=state["case_id"], step=step
        )
        operation = str(retried.get("operation", "act")).strip().lower()
        if operation != "act" or not self._has_executable_action(retried):
            return None
        return retried

    def _has_executable_action(self, decision: dict) -> bool:
        """Check whether the current protocol can execute at least one proposed action."""
        if "action" in decision:
            return True
        actions = decision.get("actions")
        return (
            int(self.config.get("max_actions_per_turn", 1)) > 1
            and isinstance(actions, list)
            and bool(actions)
        )

    def _decision_actions(self, decision: dict) -> list:
        """Return at most max_actions_per_turn actions from a model decision."""
        limit = int(self.config.get("max_actions_per_turn", 1))
        actions = decision.get("actions")
        if limit > 1 and isinstance(actions, list) and actions:
            return actions[:limit]
        if "action" in decision:
            return [decision["action"]]
        raise ValueError("Agent decision is missing action")

    def _build_messages(self, state: dict, *, turn: int = 0) -> list[dict]:
        """Build the policy input using Raw history or the PoS decision view."""
        context = self.context_provider.get_context()
        if context["type"] == "raw":
            return build_prompt(state, context["trajectory"], self.config)
        if context["type"] == "belief":
            return build_belief_prompt(state, context, self.config)
        raise ValueError(f"Unsupported context: {context['type']}")

    def _request_decision(self, messages: list[dict], case_id: str, step: int) -> dict:
        """Request JSON, logging all attempts and applying bounded format repair."""
        for attempt in range(self.config["max_json_retries"] + 1):
            self.logger.log(
                "react_model_input",
                case_id=case_id,
                step=step,
                attempt=attempt,
                messages=messages,
            )
            raw_response = self.call_model(messages)
            self.logger.log(
                "react_model_response",
                case_id=case_id,
                step=step,
                attempt=attempt,
                raw_response=raw_response,
            )
            try:
                return json.loads(raw_response)
            except json.JSONDecodeError as error:
                self.logger.log(
                    "react_model_parse_error",
                    case_id=case_id,
                    step=step,
                    attempt=attempt,
                    error=str(error),
                )
                if attempt == self.config["max_json_retries"]:
                    raise
                messages = build_repair_prompt(messages, raw_response)
