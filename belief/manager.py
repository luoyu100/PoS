"""Construct, validate, and maintain task-conditioned belief and recovery constraints."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from typing import Callable

from belief.beliefSentinel import (
    BeliefSentinel,
    ProgressFinding,
    SentinelFinding,
)
from belief.beliefStates import (
    AchievementGap,
    BeliefHealth,
    BeliefState,
    EpistemicGap,
    GapFrontier,
    Goal,
    HealthTransition,
    RecoveryDirective,
)
from belief.recoveryPlanning import compose_recovery_instruction
from belief.progressScoring import progress_outcome_signature
from belief.textualizer import build_semantic_view
from belief.trappingDetection import (
    StableGapIdentityTracker,
    detect_belief_trapping,
    gaps_semantically_equivalent,
)
from belief.worldStates import Entity, Relation, State, WorldState
from prompts.prompt_generation import (
    build_belief_repair_prompt,
    build_belief_text_prompt,
    build_belief_update_prompt,
    build_initial_belief_prompt,
)
from utils.logger import Logger


_REMOVED_CONTEXT_KEYS = {
    "evidence_validation",
    "belief_lifecycle",
    "decision_view",
    "terminal_gate",
}
_REMOVED_RECOVERY_KEYS = {
    "causal_context_max_depth",
    "allow_direct_action",
    "direct_action_min_overlap",
}


class BeliefManager:
    """Maintain B_t, the active gap, consistency checks, health, and recovery constraints."""

    def __init__(
        self,
        config: dict,
        call_model: Callable[[list[dict]], str],
        logger: Logger,
        call_sentinel: Callable[[list[dict]], str] | None = None,
        call_progress: Callable[[list[dict]], str] | None = None,
    ):
        self.config = config
        self.call_model = call_model
        self.logger = logger
        context = config.get("context", {})
        removed = sorted(_REMOVED_CONTEXT_KEYS & set(context))
        if removed:
            raise ValueError(
                "Removed non-PoS context fields are not supported: "
                + ", ".join(removed)
            )
        if context.get("agent_view", "hybrid") != "hybrid":
            raise ValueError(
                "PoS now exposes only the paper-aligned hybrid decision view"
            )
        self.task_mode = context.get("task_mode", "execution")
        self.update_every_steps = int(context.get("update_every_steps", 1))
        self.diagnostic_threshold = float(
            context.get("diagnostic_progress_threshold", 0.3)
        )
        model_config = context.get("model", {})
        self.max_json_retries = min(
            1,
            int(model_config.get("max_json_retries", 1)),
        )

        self.sentinel_config = context.get("sentinel", {})
        sentinel_enabled = bool(self.sentinel_config.get("enabled", False))
        self.sentinel_consistency_enabled = sentinel_enabled and bool(
            self.sentinel_config.get("consistency_enabled", True)
        )
        self.sentinel_progress_enabled = sentinel_enabled and bool(
            self.sentinel_config.get("progress_enabled", True)
        )
        if self.sentinel_consistency_enabled and call_sentinel is None:
            raise ValueError("PoS consistency Sentinel requires a model caller")
        if (
            self.sentinel_progress_enabled
            and self.task_mode == "execution"
            and call_progress is None
        ):
            raise ValueError("Execution progress scoring requires a model caller")
        sentinel_runtime_config = dict(self.sentinel_config)
        sentinel_runtime_config["consistency_enabled"] = (
            self.sentinel_consistency_enabled
        )
        sentinel_runtime_config["progress_enabled"] = self.sentinel_progress_enabled
        self.sentinel = (
            BeliefSentinel(
                sentinel_runtime_config,
                call_sentinel,
                logger,
                call_progress_model=call_progress,
            )
            if sentinel_enabled
            else None
        )
        self.local_audit_every = int(
            self.sentinel_config.get("local_audit_every_steps", 1)
        )
        self.global_audit_every = int(
            self.sentinel_config.get("global_audit_every_steps", 8)
        )
        self.recent_transition_count = int(
            self.sentinel_config.get("recent_transition_count", 4)
        )

        self.trapping_config = context.get("trapping", {})
        self.recovery_config = context.get("recovery", {})
        removed_recovery = sorted(_REMOVED_RECOVERY_KEYS & set(self.recovery_config))
        if removed_recovery:
            raise ValueError(
                "Removed non-PoS recovery fields are not supported: "
                + ", ".join(removed_recovery)
            )
        self.case_id = ""
        self.belief_state: BeliefState | None = None
        self.raw_trajectory: list[dict] = []
        self.pending_issues: list[SentinelFinding] = []
        self.belief_update_history: list[dict] = []
        self.health_transitions: list[HealthTransition] = []
        self.health_history: list[dict] = []
        self.belief_snapshots: list[dict] = []
        self.frontier_episode = 0
        self.frontier_history: list[dict] = []
        self.active_recovery: RecoveryDirective | None = None
        self.recovery_history: list[dict] = []
        self.task_done = False
        self.task_success = False
        self._gap_identity_tracker = StableGapIdentityTracker()

    def reset(self, state: dict) -> None:
        """Initialize B_0 and the first active gap from the task and observation."""

        self.case_id = state["case_id"]
        self.raw_trajectory = [
            {
                "step": -1,
                "observation": state["observation"],
                "done": state["done"],
                "success": state["success"],
                "score": state["score"],
                "available_actions": state["actions"],
            }
        ]
        self.pending_issues = []
        self.belief_update_history = []
        self.health_transitions = []
        self.health_history = []
        self.belief_snapshots = []
        self.frontier_episode = 0
        self.frontier_history = []
        self.active_recovery = None
        self.recovery_history = []
        self.task_done = bool(state["done"])
        self.task_success = bool(state["success"])
        self._gap_identity_tracker.reset()

        base = BeliefState(
            world=WorldState(),
            goal=Goal(user_input=state["task"], specification=""),
        )
        initial_messages = build_initial_belief_prompt(state, self.config)
        update, candidate = self._request_candidate(
            initial_messages,
            base=base,
            operation="initialize",
            step=-1,
        )
        base.goal.specification = str(update["goal_specification"])
        final = self._validate_candidate(
            base=base,
            candidate=candidate,
            transition={
                "thought": "",
                "action": None,
                "observation": state["observation"],
            },
            state=state,
            step=-1,
            initial_update=update,
        )
        self._record_belief_update(
            step=-1,
            base=base,
            candidate=candidate,
            final=final,
        )
        self.belief_state = final
        self._commit_frontier_change(None, final.frontier, step=-1)
        self._reconcile_gap_identities()
        self._record_snapshot(step=-1)
        self._refresh_belief_text(step=-1)
        self.logger.log(
            "belief_state_initialized",
            case_id=self.case_id,
            step=-1,
            belief_state=self.get_context(),
            entities=len(final.world.entities),
            states=len(final.world.states),
            relations=len(final.world.relations),
            epistemic_gaps=len(final.epistemic_gaps),
            achievement_gaps=len(final.achievement_gaps),
        )

    def update(self, transition: dict, state: dict, step: int) -> None:
        """Process one actual action-observation transition at the preserved update granularity."""

        belief = self._current()
        self.task_done = bool(state["done"])
        self.task_success = bool(state["success"])
        self.raw_trajectory.append(
            {
                "step": step,
                "action": transition["action"],
                "observation": transition["observation"],
                "done": state["done"],
                "success": state["success"],
                "score": state["score"],
                "available_actions": state["actions"],
            }
        )
        belief.advance_step()
        if (
            self.active_recovery is not None
            and step >= self.active_recovery.applies_from_step
        ):
            self.active_recovery.recovery_action_steps.append(step)

        if (step + 1) % self.update_every_steps != 0:
            return
        self._update_belief(copy.deepcopy(transition), state, step)

    def _update_belief(
        self,
        transition: dict,
        state: dict,
        step: int,
    ) -> None:
        """Process candidate construction, validation, progress, and health in order."""

        belief = self._current()
        before_state = copy.deepcopy(belief)
        belief_before = self._context_from_state(before_state)
        messages = build_belief_update_prompt(
            belief_context=belief_before,
            transition=transition,
            state=state,
            config=self.config,
            step=step,
        )
        update, candidate = self._request_candidate(
            messages,
            base=before_state,
            operation="update",
            step=step,
        )
        self.logger.log(
            "belief_candidate_generated",
            case_id=self.case_id,
            step=step,
            candidate_belief=self._context_from_state(candidate),
        )
        final = self._validate_candidate(
            base=before_state,
            candidate=candidate,
            transition=transition,
            state=state,
            step=step,
            initial_update=update,
        )
        self._record_belief_update(
            step=step,
            base=before_state,
            candidate=candidate,
            final=final,
        )

        previous_frontier = before_state.frontier
        transition_frontier_episode = self.frontier_episode
        self.belief_state = final
        self._commit_frontier_change(previous_frontier, final.frontier, step=step)
        self._reconcile_gap_identities()
        belief_after = self._structured_context()
        transition_record = HealthTransition(
            step=step,
            frontier_episode=transition_frontier_episode,
            frontier=copy.deepcopy(previous_frontier),
            action=transition["action"],
            observation=transition["observation"],
            done=state["done"],
            success=state["success"],
            task_score=float(state["score"]),
        )
        self.health_transitions.append(transition_record)
        self._record_snapshot(step=step)

        if self.sentinel is not None and self.sentinel_progress_enabled:
            self.sentinel.submit_progress(
                case_id=self.case_id,
                step=step,
                frontier_episode=transition_frontier_episode,
                frontier=(
                    asdict(previous_frontier) if previous_frontier is not None else None
                ),
                action=transition["action"],
                observation=transition["observation"],
                task_mode=self.task_mode,
                belief_before=belief_before,
                belief_after=belief_after,
                diagnostic_threshold=self.diagnostic_threshold,
                frontier_history=self._recent_progress_history(previous_frontier),
                done=state["done"],
                success=state["success"],
                task_score=float(state["score"]),
            )
            findings = self.sentinel.collect_progress_ready()
            if findings:
                self._apply_progress_finding(findings[-1])

        if self.task_done:
            self._finish_recovery("task_terminal")
        else:
            self._refresh_belief_text(step=step)
        self.logger.log(
            "belief_state_updated",
            case_id=self.case_id,
            step=step,
            belief_state=self.get_context(),
            entities=len(final.world.entities),
            states=len(final.world.states),
            relations=len(final.world.relations),
            epistemic_gaps=len(final.epistemic_gaps),
            achievement_gaps=len(final.achievement_gaps),
        )

    def get_context(self) -> dict:
        """Return the structured belief and derived text used for the next policy decision."""

        context = self._structured_context()
        context["belief_text"] = self._current().belief_text
        if self._recovery_is_active():
            context["recovery"] = self._recovery_context()
        return context

    def get_result(self) -> dict:
        """Export belief history, progress, health, frontier, and recovery events."""

        result = {
            "belief_state": {
                **self._structured_context(),
                "belief_text": self._current().belief_text,
            },
            "raw_trajectory": self.raw_trajectory,
            "belief_update_history": self.belief_update_history,
            "belief_health": asdict(self._current().health),
            "belief_health_history": self.health_history,
            "health_transitions": [
                asdict(transition) for transition in self.health_transitions
            ],
            "frontier_history": self.frontier_history,
            "active_recovery": (
                asdict(self.active_recovery)
                if self.active_recovery is not None
                else None
            ),
            "recovery_history": self.recovery_history,
        }
        if self.sentinel is not None:
            result["belief_sentinel"] = {
                "pending_issues": [asdict(finding) for finding in self.pending_issues],
                "audit_history": self.sentinel.audit_history,
                "progress_history": self.sentinel.progress_history,
            }
        return result

    def close(self) -> None:
        """Finish the synchronous Sentinel when an episode exits early."""

        if self.sentinel is not None and not self.sentinel.closed:
            self.pending_issues.extend(self.sentinel.finish())

    def _validate_candidate(
        self,
        *,
        base: BeliefState,
        candidate: BeliefState,
        transition: dict,
        state: dict,
        step: int,
        initial_update: dict,
    ) -> BeliefState:
        """Audit before commit; retain the previous belief if the repair also fails."""

        mode, evidence = self._audit_scope(step)
        if (
            mode is None
            or self.sentinel is None
            or not self.sentinel_consistency_enabled
        ):
            self.logger.log(
                "belief_candidate_validated",
                case_id=self.case_id,
                step=step,
                audit_mode=None,
            )
            return candidate

        before = self._context_from_state(base)
        try:
            issues = self.sentinel.audit_candidate(
                case_id=self.case_id,
                base_step=step,
                mode=mode,
                belief_before=before,
                candidate_belief=self._context_from_state(candidate),
                raw_trajectory=evidence,
            )
        except RuntimeError as error:
            self.logger.log(
                "belief_candidate_rejected",
                case_id=self.case_id,
                step=step,
                error=str(error),
            )
            return base
        if not issues:
            self.logger.log(
                "belief_candidate_validated",
                case_id=self.case_id,
                step=step,
                audit_mode=mode,
            )
            return candidate

        issue_payload = [asdict(finding) for finding in issues]
        repair_messages = build_belief_update_prompt(
            belief_context=before,
            transition=transition,
            state=state,
            config=self.config,
            consistency_issues=issue_payload,
            step=step,
        )
        repaired_update, repaired = self._request_candidate(
            repair_messages,
            base=base,
            operation="sentinel_repair",
            step=step,
        )
        try:
            remaining = self.sentinel.audit_candidate(
                case_id=self.case_id,
                base_step=step,
                mode=mode,
                belief_before=before,
                candidate_belief=self._context_from_state(repaired),
                raw_trajectory=evidence,
            )
        except RuntimeError as error:
            self.logger.log(
                "belief_candidate_rejected",
                case_id=self.case_id,
                step=step,
                error=str(error),
            )
            return base
        if remaining:
            self.pending_issues.extend(remaining)
            self.logger.log(
                "belief_candidate_rejected",
                case_id=self.case_id,
                step=step,
                issues=[asdict(finding) for finding in remaining],
            )
            return base
        self.logger.log(
            "belief_candidate_repaired",
            case_id=self.case_id,
            step=step,
            issues=issue_payload,
        )
        return repaired

    def _audit_scope(
        self,
        step: int,
    ) -> tuple[str | None, list[dict]]:
        """Select global auditing at its interval instead of running both local and global."""

        if step == -1:
            return "local", list(self.raw_trajectory)
        update_number = 1 + sum(
            int(record["step"] >= 0) for record in self.belief_update_history
        )
        if self.global_audit_every > 0 and update_number % self.global_audit_every == 0:
            return "global", list(self.raw_trajectory)
        if self.local_audit_every > 0 and update_number % self.local_audit_every == 0:
            return "local", self.raw_trajectory[-self.recent_transition_count :]
        return None, []

    @staticmethod
    def _stabilize_gap_targets(
        previous: list[EpistemicGap] | list[AchievementGap],
        proposed: list[dict],
        *,
        gap_type: str,
    ) -> list[dict]:
        """Preserve the original target of an equivalent unresolved gap."""

        stabilized = []
        used: set[int] = set()
        for raw_gap in proposed:
            gap = dict(raw_gap)
            matches = [
                index
                for index, old_gap in enumerate(previous)
                if index not in used
                and gaps_semantically_equivalent(
                    old_gap,
                    gap,
                    gap_type=gap_type,
                )
            ]
            if len(matches) == 1:
                index = matches[0]
                gap["target"] = previous[index].target
                used.add(index)
            stabilized.append(gap)
        return stabilized

    def _build_candidate(
        self,
        base: BeliefState,
        update: dict,
    ) -> BeliefState:
        """Apply a delta to a deep copy without mutating the committed belief."""

        candidate = copy.deepcopy(base)
        self._apply_world_update(candidate.world, update)
        epistemic_gaps = self._stabilize_gap_targets(
            base.epistemic_gaps,
            update.get("epistemic_gaps", []),
            gap_type="epistemic",
        )
        candidate.epistemic_gaps = [EpistemicGap(**gap) for gap in epistemic_gaps]
        stable_achievement_gaps = self._stabilize_gap_targets(
            base.achievement_gaps,
            update.get("achievement_gaps", []),
            gap_type="achievement",
        )
        achievement_gaps = [AchievementGap(**gap) for gap in stable_achievement_gaps]
        candidate.achievement_gaps = achievement_gaps
        primary = self._select_frontier(
            candidate,
            previous=base.frontier,
            proposed=update.get("frontier"),
        )
        candidate.frontier = primary
        return candidate

    def _record_belief_update(
        self,
        *,
        step: int,
        base: BeliefState,
        candidate: BeliefState,
        final: BeliefState,
    ) -> None:
        """Record both candidate and finalized belief for result export."""

        status = (
            "validated"
            if final == candidate
            else "rejected"
            if final == base
            else "repaired"
        )
        self.belief_update_history.append(
            {
                "step": step,
                "status": status,
                "candidate_belief": self._context_from_state(candidate),
                "final_belief": self._context_from_state(final),
            }
        )

    @staticmethod
    def _apply_world_update(world: WorldState, update: dict) -> None:
        """Delete relations and states before upserting entities, states, and relations."""

        for relation_id in update.get("remove_relation_ids", []):
            world.remove_relation(relation_id)
        for state_id in update.get("remove_state_ids", []):
            world.remove_state(state_id)
        for data in update.get("upsert_entities", update.get("entities", [])):
            world.upsert_entity(Entity(**data))
        for data in update.get("upsert_states", update.get("states", [])):
            world.upsert_state(State(**data))
        for data in update.get("upsert_relations", update.get("relations", [])):
            world.upsert_relation(Relation(**data))

    def _select_frontier(
        self,
        belief: BeliefState,
        *,
        previous: GapFrontier | None,
        proposed: dict | None,
    ) -> GapFrontier | None:
        """Keep an unresolved frontier; otherwise select a valid proposed or first gap."""

        if previous is not None and self._find_gap_in_state(belief, previous):
            return copy.deepcopy(previous)
        if isinstance(proposed, dict):
            candidate = GapFrontier(
                gap_type=proposed["gap_type"],
                target=proposed["target"],
            )
            gap = self._find_gap_in_state(belief, candidate)
            if gap is not None:
                return GapFrontier(candidate.gap_type, gap.target)
        if belief.epistemic_gaps:
            return GapFrontier("epistemic", belief.epistemic_gaps[0].target)
        if belief.achievement_gaps:
            return GapFrontier("achievement", belief.achievement_gaps[0].target)
        return None

    def _commit_frontier_change(
        self,
        previous: GapFrontier | None,
        current: GapFrontier | None,
        *,
        step: int,
    ) -> None:
        """Record a frontier change only after the candidate is committed."""

        if self._frontiers_equivalent(previous, current):
            return
        self.frontier_episode += 1
        event = {
            "step": step,
            "frontier_episode": self.frontier_episode,
            "previous": asdict(previous) if previous is not None else None,
            "current": asdict(current) if current is not None else None,
            "current_gap": self._frontier_context(),
        }
        self.frontier_history.append(event)
        self.logger.log("belief_frontier_updated", case_id=self.case_id, **event)

    @staticmethod
    def _frontiers_equivalent(
        left: GapFrontier | None,
        right: GapFrontier | None,
    ) -> bool:
        """Check whether two frontier pointers identify the same semantic gap."""

        if left is None or right is None:
            return left is right
        return left.gap_type == right.gap_type and gaps_semantically_equivalent(
            left,
            right,
            gap_type=left.gap_type,
        )

    def _find_gap_in_state(
        self,
        belief: BeliefState,
        frontier: GapFrontier,
    ) -> EpistemicGap | AchievementGap | None:
        """Resolve a frontier pointer in a specified belief state."""

        gaps = (
            belief.epistemic_gaps
            if frontier.gap_type == "epistemic"
            else belief.achievement_gaps
        )
        matches = [
            gap
            for gap in gaps
            if gaps_semantically_equivalent(
                gap,
                frontier,
                gap_type=frontier.gap_type,
            )
        ]
        return matches[0] if len(matches) == 1 else None

    def _frontier_context(self) -> dict | None:
        """Expose the full active gap; the pointer does not duplicate its reason."""

        frontier = self._current().frontier
        if frontier is None:
            return None
        gap = self._find_gap_in_state(self._current(), frontier)
        if gap is None:
            return None
        return {
            "gap_type": frontier.gap_type,
            "target": gap.target,
            "reason": gap.reason,
        }

    def _reconcile_gap_identities(self) -> None:
        """Advance stable gap identities only after a validated belief commit."""

        belief = self._current()
        self._gap_identity_tracker.reconcile(
            "epistemic",
            belief.epistemic_gaps,
        )
        self._gap_identity_tracker.reconcile(
            "achievement",
            belief.achievement_gaps,
        )

    def _record_snapshot(self, *, step: int) -> None:
        """Store a validated belief snapshot for the health window."""

        self.belief_snapshots.append(
            {
                "step": step,
                "world": self._current().world.to_dict(),
                "epistemic_gap_ids": [
                    gap.identity
                    for gap in self._gap_identity_tracker.active("epistemic")
                ],
                "achievement_gap_ids": [
                    gap.identity
                    for gap in self._gap_identity_tracker.active("achievement")
                ],
                "frontier": (
                    asdict(self._current().frontier)
                    if self._current().frontier is not None
                    else None
                ),
            }
        )

    def _apply_progress_finding(self, finding: ProgressFinding) -> None:
        """Bind u_t to its action and step, then update health and recovery."""

        transition = next(
            item
            for item in reversed(self.health_transitions)
            if item.step == finding.step
        )
        transition.progress = finding.progress
        transition.scoring_method = finding.scoring_method
        transition.score = finding.score
        transition.affected_record_ids = finding.affected_record_ids
        transition.reason = finding.reason
        self.logger.log(
            "belief_transition_progress_applied",
            case_id=self.case_id,
            step=finding.step,
            action=finding.action,
            progress=finding.progress,
            score=finding.score,
            reason=finding.reason,
        )
        if self.task_done:
            self._finish_recovery("task_terminal")
        else:
            self._update_health()

    def _update_health(self) -> None:
        """Compute persistence, stagnation, recurrence, and health over the window."""

        if not self.trapping_config.get("enabled", False):
            return
        detection = detect_belief_trapping(
            self.health_transitions,
            self.belief_snapshots,
            active_gap=self._frontier_context(),
            window_size=int(self.trapping_config.get("window_size", 8)),
            health_threshold=float(self.trapping_config.get("health_threshold", 0.25)),
            cycle_threshold=float(self.trapping_config.get("cycle_threshold", 0.75)),
            recurrence_epsilon=float(
                self.trapping_config.get("recurrence_epsilon", 0.15)
            ),
            max_cycle_length=int(self.trapping_config.get("max_cycle_length", 4)),
        )
        self._current().health = BeliefHealth(**detection.to_belief_health_kwargs())
        self.health_history.append(
            {
                "step": self._current().step,
                **asdict(self._current().health),
            }
        )
        self.logger.log(
            "belief_health_updated",
            case_id=self.case_id,
            step=self._current().step,
            health=asdict(self._current().health),
        )
        self._update_recovery()

    def _update_recovery(self) -> None:
        """Release resolved or no-longer-trapped constraints; start recovery for new trapping."""

        health = self._current().health
        if not self.recovery_config.get("enabled", False) or not health.trapped:
            self._finish_recovery("health_recovered")
            return
        if self.active_recovery is not None:
            self._finish_recovery("rediagnosed")
        self._start_recovery()

    def _start_recovery(self) -> None:
        """Select a gap in the trapped dimension and combine gap and pattern constraints."""

        health = self._current().health
        gap_type = health.gap_dimension
        pattern = health.trapping_pattern
        if gap_type is None or self._current().frontier is None:
            return
        gaps = (
            self._current().epistemic_gaps
            if gap_type == "epistemic"
            else self._current().achievement_gaps
        )
        if not gaps:
            return
        persistent_ids = (
            health.persistent_epistemic_gaps
            if gap_type == "epistemic"
            else health.persistent_achievement_gaps
        )
        if persistent_ids:
            persistent_targets = {
                gap.target
                for gap in self._gap_identity_tracker.active(gap_type)
                if gap.identity in persistent_ids
            }
            gaps = [gap for gap in gaps if gap.target in persistent_targets] or gaps
        active = self._current().frontier
        selected = next(
            (
                gap
                for gap in gaps
                if active.gap_type == gap_type and active.target == gap.target
            ),
            gaps[0],
        )
        frontier = GapFrontier(gap_type, selected.target)
        recent_count = int(self.recovery_config.get("recent_low_progress_count", 4))
        low_progress = [
            {
                "step": item.step,
                "action": item.action,
                "progress": item.progress,
                "score": item.score,
                "reason": item.reason,
            }
            for item in self.health_transitions
            if item.progress == 0
        ][-recent_count:]
        instruction = compose_recovery_instruction(
            pattern=pattern,
            gap_type=gap_type,
            gap={"target": selected.target, "reason": selected.reason},
            recent_transitions=[
                {"step": item.step, "action": item.action, "progress": item.progress}
                for item in self.health_transitions
                if item.progress in (0, 1)
            ][-recent_count:],
            recurrence_period=health.recurrence_period,
            cycle_transitions=self._cycle_transitions(),
        )
        self.active_recovery = RecoveryDirective(
            recovery_frontier=frontier,
            trigger_frontier=copy.deepcopy(self._current().frontier),
            gap_dimension=gap_type,
            trapping_pattern=pattern,
            started_at_step=self._current().step,
            applies_from_step=self._current().step,
            low_progress_transitions=low_progress,
            instruction=instruction,
        )
        self.logger.log(
            "belief_recovery_started",
            case_id=self.case_id,
            step=self._current().step,
            recovery=asdict(self.active_recovery),
        )

    def _cycle_transitions(self) -> list[dict]:
        """Expose validated state-action transitions matching the dominant lag."""
        from belief.trappingDetection import (
            _active_gap_projection,
            _projection_distance,
        )

        period = self._current().health.recurrence_period
        if self._current().health.trapping_pattern != "cycle" or period is None:
            return []
        window = [item for item in self.health_transitions if item.progress in (0, 1)][
            -int(self.trapping_config.get("window_size", 8)) :
        ]
        snapshots = {item["step"]: item for item in self.belief_snapshots}
        epsilon = float(self.trapping_config.get("recurrence_epsilon", 0.15))
        focus = self._frontier_context()
        for index in range(len(window) - 1, period - 1, -1):
            now, previous = window[index], window[index - period]
            if (
                _projection_distance(
                    _active_gap_projection(snapshots[now.step]["world"], focus),
                    _active_gap_projection(snapshots[previous.step]["world"], focus),
                )
                <= epsilon
            ):
                return [
                    {
                        "step": item.step,
                        "action": item.action,
                        "progress": item.progress,
                        "world_before": snapshots[
                            max(s for s in snapshots if s < item.step)
                        ]["world"],
                        "world_after": snapshots[item.step]["world"],
                    }
                    for j in range(index - period + 1, index + 1)
                    for item in [window[j]]
                ]
        return []

    def _finish_recovery(self, status: str) -> None:
        """Record the recovery termination reason and clear the temporary constraint."""

        if self.active_recovery is None:
            return
        event = {
            "status": status,
            "step": self._current().step,
            "recovery": asdict(self.active_recovery),
        }
        self.recovery_history.append(event)
        self.logger.log("belief_recovery_finished", case_id=self.case_id, **event)
        self.active_recovery = None

    def _recovery_is_active(self) -> bool:
        """Expose recovery only once its applies_from_step has been reached."""

        return (
            self.active_recovery is not None
            and self._current().step >= self.active_recovery.applies_from_step
        )

    def _recovery_context(self) -> dict:
        """Return recovery constraints and their target gap."""

        recovery = self.active_recovery
        assert recovery is not None
        gap = self._find_gap_in_state(
            self._current(),
            recovery.recovery_frontier,
        )
        payload = asdict(recovery)
        payload["recovery_gap"] = (
            {
                "gap_type": recovery.recovery_frontier.gap_type,
                "target": gap.target,
                "reason": gap.reason,
            }
            if gap is not None
            else None
        )
        return payload

    def _recent_progress_history(
        self,
        frontier: GapFrontier | None,
    ) -> list[dict]:
        """Collect recent action-outcome scores under the same decision frontier."""

        return [
            {
                "step": item.step,
                "verified_progress": item.progress,
                "action": item.action,
                "outcome_signature": progress_outcome_signature(item.observation),
            }
            for item in self.health_transitions
            if item.progress in (0, 1)
            and self._frontiers_equivalent(item.frontier, frontier)
        ][-self.recent_transition_count :]

    def _refresh_belief_text(self, *, step: int) -> None:
        """Generate a compact text projection from the finalized structured belief."""

        belief = self._current()
        messages = build_belief_text_prompt(
            semantic_view=build_semantic_view(belief.world),
            goal=asdict(belief.goal),
            epistemic_gaps=[asdict(gap) for gap in belief.epistemic_gaps],
            achievement_gaps=[asdict(gap) for gap in belief.achievement_gaps],
            config=self.config,
            frontier_gap=self._frontier_context(),
        )
        response = self._request_json(
            messages,
            operation="textualize",
            step=step,
        )
        belief.belief_text = str(response["belief_text"])
        self.logger.log(
            "belief_text_updated",
            case_id=self.case_id,
            step=step,
            belief_text=belief.belief_text,
        )

    def _request_json(
        self,
        messages: list[dict[str, str]],
        *,
        operation: str,
        step: int,
    ) -> dict:
        """Log model input and output and apply the configured bounded JSON repair."""

        for attempt in range(self.max_json_retries + 1):
            self.logger.log(
                "belief_model_input",
                case_id=self.case_id,
                step=step,
                operation=operation,
                attempt=attempt,
                messages=messages,
            )
            raw = self.call_model(messages)
            self.logger.log(
                "belief_model_response",
                case_id=self.case_id,
                step=step,
                operation=operation,
                attempt=attempt,
                raw_response=raw,
            )
            try:
                response = json.loads(raw)
                response = self._normalize_model_payload(response, operation)
                self._validate_model_payload(response, operation)
                return response
            except json.JSONDecodeError as error:
                self.logger.log(
                    "belief_model_parse_error",
                    case_id=self.case_id,
                    step=step,
                    operation=operation,
                    attempt=attempt,
                    error=str(error),
                )
                if attempt == self.max_json_retries:
                    raise
                messages = build_belief_repair_prompt(messages, raw)
            except (KeyError, TypeError, ValueError) as error:
                self.logger.log(
                    "belief_model_schema_error",
                    case_id=self.case_id,
                    step=step,
                    operation=operation,
                    attempt=attempt,
                    error=str(error),
                )
                if attempt == self.max_json_retries:
                    raise
                messages = build_belief_repair_prompt(
                    messages,
                    raw,
                    schema_error=str(error),
                )
        raise RuntimeError("Belief model did not return JSON")

    @staticmethod
    def _normalize_model_payload(response: object, operation: str) -> object:
        """Map supported relation-field aliases to canonical schema fields."""

        if not isinstance(response, dict) or operation == "textualize":
            return response
        relation_key = "relations" if operation == "initialize" else "upsert_relations"
        relations = response.get(relation_key)
        if not isinstance(relations, list):
            return response
        aliases = {
            "source_entity_id": "source_id",
            "target_entity_id": "target_id",
            "src_entity_id": "source_id",
            "dst_entity_id": "target_id",
            "source": "source_id",
            "target": "target_id",
            "src": "source_id",
            "dst": "target_id",
            "relation_type": "description",
            "relation": "description",
            "type": "description",
        }
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            for alias, canonical in aliases.items():
                if alias not in relation:
                    continue
                relation.setdefault(canonical, relation[alias])
                del relation[alias]
        return response

    def _request_candidate(
        self,
        messages: list[dict[str, str]],
        *,
        base: BeliefState,
        operation: str,
        step: int,
    ) -> tuple[dict, BeliefState]:
        """Build a candidate, allowing one targeted repair of cross-record references."""

        update = self._request_json(messages, operation=operation, step=step)
        if operation == "initialize":
            base.goal.specification = str(update["goal_specification"])
        try:
            return update, self._build_candidate(base, update)
        except (KeyError, TypeError, ValueError) as error:
            self.logger.log(
                "belief_candidate_schema_error",
                case_id=self.case_id,
                step=step,
                operation=operation,
                error=str(error),
            )
            repair_messages = build_belief_repair_prompt(
                messages,
                json.dumps(update, ensure_ascii=False),
                schema_error=str(error),
            )
            repaired_update = self._request_json(
                repair_messages,
                operation=operation,
                step=step,
            )
            if operation == "initialize":
                base.goal.specification = str(repaired_update["goal_specification"])
            return repaired_update, self._build_candidate(base, repaired_update)

    @staticmethod
    def _validate_model_payload(response: object, operation: str) -> None:
        """Check model-output structure before committing the candidate."""

        if not isinstance(response, dict):
            raise TypeError("top-level value must be an object")
        if operation == "textualize":
            if set(response) != {"belief_text"} or not isinstance(
                response["belief_text"],
                str,
            ):
                raise ValueError("textualize must return only string belief_text")
            return

        initial = operation == "initialize"
        expected_keys = (
            {
                "goal_specification",
                "entities",
                "states",
                "relations",
                "epistemic_gaps",
                "achievement_gaps",
                "frontier",
            }
            if initial
            else {
                "remove_state_ids",
                "remove_relation_ids",
                "upsert_entities",
                "upsert_states",
                "upsert_relations",
                "epistemic_gaps",
                "achievement_gaps",
                "frontier",
            }
        )
        if set(response) != expected_keys:
            missing = sorted(expected_keys - set(response))
            extra = sorted(set(response) - expected_keys)
            raise ValueError(f"top-level missing={missing}, extra={extra}")
        if initial and not isinstance(response["goal_specification"], str):
            raise TypeError("goal_specification must be a string")

        if not initial:
            for key in ("remove_state_ids", "remove_relation_ids"):
                values = response[key]
                if not isinstance(values, list) or not all(
                    isinstance(value, str) for value in values
                ):
                    raise TypeError(f"{key} must be an array of strings")

        record_specs = (
            (
                ("entities", Entity),
                ("states", State),
                ("relations", Relation),
            )
            if initial
            else (
                ("upsert_entities", Entity),
                ("upsert_states", State),
                ("upsert_relations", Relation),
            )
        )
        for key, record_type in record_specs + (
            ("epistemic_gaps", EpistemicGap),
            ("achievement_gaps", AchievementGap),
        ):
            records = response[key]
            if not isinstance(records, list):
                raise TypeError(f"{key} must be an array")
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    raise TypeError(f"{key}[{index}] must be a non-null object")
                try:
                    record_type(**record)
                except TypeError as error:
                    raise ValueError(f"{key}[{index}]: {error}") from error

        frontier = response["frontier"]
        if frontier is not None:
            if not isinstance(frontier, dict):
                raise TypeError("frontier must be an object or null")
            try:
                GapFrontier(**frontier)
            except TypeError as error:
                raise ValueError(f"frontier: {error}") from error

    def _structured_context(self) -> dict:
        """Export structured belief without treating derived belief text as a fact source."""

        return self._context_from_state(self._current())

    def _context_from_state(self, belief: BeliefState) -> dict:
        """Convert any belief state to its serializable context representation."""

        frontier_gap = None
        if belief.frontier is not None:
            gap = self._find_gap_in_state(belief, belief.frontier)
            if gap is not None:
                frontier_gap = {
                    "gap_type": belief.frontier.gap_type,
                    "target": gap.target,
                    "reason": gap.reason,
                }
        return {
            "type": "belief",
            "goal": asdict(belief.goal),
            "world": belief.world.to_dict(),
            "epistemic_gaps": [asdict(gap) for gap in belief.epistemic_gaps],
            "achievement_gaps": [asdict(gap) for gap in belief.achievement_gaps],
            "health": asdict(belief.health),
            "frontier": (
                asdict(belief.frontier) if belief.frontier is not None else None
            ),
            "frontier_gap": frontier_gap,
            "step": belief.step,
        }

    def _current(self) -> BeliefState:
        """Return the initialized belief state."""

        if self.belief_state is None:
            raise RuntimeError("BeliefManager has not been initialized")
        return self.belief_state
