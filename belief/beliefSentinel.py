"""Synchronous candidate consistency auditing and validated-transition scoring."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

from belief.progressScoring import (
    diagnostic_progress,
)
from prompts.beliefSentinel_prompt import (
    build_execution_progress_prompt,
    build_global_audit_prompt,
    build_local_audit_prompt,
    build_progress_repair_prompt,
    build_sentinel_repair_prompt,
)
from utils.logger import Logger


@dataclass(slots=True)
class SentinelIssue:
    """Minimal belief correction supported by quoted trajectory evidence."""

    targets: list[str]
    issue_type: str
    problem: str
    evidence_step: int
    evidence_quote: str
    evidence: str
    suggestion: str


@dataclass(slots=True)
class SentinelFinding:
    """Consistency issues together with the audited scope."""

    base_step: int
    audit_mode: Literal["local", "global"]
    issue: SentinelIssue


@dataclass(slots=True)
class ProgressFinding:
    """Binary progress u_t for one validated transition."""

    step: int
    frontier_episode: int
    frontier: dict | None
    action: Any
    observation: str
    done: bool
    success: bool
    task_score: float
    progress: int
    scoring_method: str
    score: float
    affected_record_ids: tuple[str, ...]
    reason: str


class BeliefSentinel:
    """Audit consistency before scoring progress within the same agent step."""

    def __init__(
        self,
        config: dict,
        call_model: Callable[[list[dict]], str] | None,
        logger: Logger,
        call_progress_model: Callable[[list[dict]], str] | None = None,
    ):
        self.config = config
        self.call_model = call_model
        self.call_progress_model = call_progress_model
        self.logger = logger
        self.consistency_enabled = bool(config.get("consistency_enabled", True))
        self.progress_enabled = bool(config.get("progress_enabled", True))
        consistency_model = config.get("consistency_model") or config.get("model", {})
        progress_model = config.get("progress_model") or consistency_model
        self.max_json_retries = min(
            1,
            int(consistency_model.get("max_json_retries", 1)),
        )
        self.progress_max_json_retries = min(
            1,
            int(progress_model.get("max_json_retries", 1)),
        )
        self.audit_observation_max_chars = int(
            config.get("audit_observation_max_chars", 6000)
        )
        self.ready_issues: list[SentinelFinding] = []
        self.ready_progress: list[ProgressFinding] = []
        self.audit_history: list[dict] = []
        self.progress_history: list[dict] = []
        self.closed = False

    def audit_candidate(
        self,
        *,
        case_id: str,
        base_step: int,
        mode: Literal["local", "global"],
        belief_before: dict,
        candidate_belief: dict,
        raw_trajectory: list[dict],
    ) -> list[SentinelFinding]:
        """Audit an uncommitted candidate; global auditing expands the evidence window."""

        if not self.consistency_enabled:
            return []
        compact = self._compact_trajectory(raw_trajectory)
        audit_input = {
            "step": base_step,
            "belief_before": belief_before,
            "candidate_belief": candidate_belief,
            "changed_ids": self._changed_record_ids(
                belief_before.get("world", {}),
                candidate_belief.get("world", {}),
            ),
            "raw_trajectory": compact,
        }
        messages = (
            build_global_audit_prompt(audit_input)
            if mode == "global"
            else build_local_audit_prompt(audit_input)
        )
        self.logger.log(
            "belief_sentinel_audit_submitted",
            case_id=case_id,
            step=base_step,
            audit_mode=mode,
        )
        payload = self._request_json(
            messages,
            self.call_model,
            self.max_json_retries,
            case_id=case_id,
            step=base_step,
            lane="consistency",
            audit_mode=mode,
            repair=build_sentinel_repair_prompt,
        )
        if payload is None:
            self.logger.log(
                "belief_sentinel_audit_failed",
                case_id=case_id,
                step=base_step,
                audit_mode=mode,
                error="invalid model response",
            )
            raise RuntimeError("Sentinel consistency validation failed")

        evidence_by_step = {
            int(item["step"]): self._trajectory_text(item) for item in compact
        }
        valid_targets = self._valid_targets(belief_before) | self._valid_targets(
            candidate_belief
        )
        raw_issues = payload.get("issues")
        if not isinstance(raw_issues, list):
            raise RuntimeError("Sentinel response must contain an issues list")
        issues: list[SentinelIssue] = []
        for data in raw_issues[:1]:
            try:
                issue = SentinelIssue(**data)
            except TypeError:
                raise RuntimeError("Sentinel issue does not match its schema")
            if self._valid_issue(issue, valid_targets, evidence_by_step):
                issues.append(issue)
            else:
                self.logger.log(
                    "belief_sentinel_issue_discarded",
                    case_id=case_id,
                    step=base_step,
                    audit_mode=mode,
                    reason="issue target or evidence is not grounded",
                    issue=asdict(issue),
                )
        findings = [SentinelFinding(base_step, mode, issue) for issue in issues]
        self.audit_history.append(
            {
                "step": base_step,
                "audit_mode": mode,
                "issues": [asdict(item) for item in findings],
            }
        )
        self.logger.log(
            "belief_sentinel_audit_completed",
            case_id=case_id,
            step=base_step,
            audit_mode=mode,
            issue_count=len(findings),
            issues=[asdict(item.issue) for item in findings],
        )
        return findings

    def submit_local(
        self,
        case_id: str,
        base_step: int,
        belief_before: dict,
        belief_after: dict,
        recent_trajectory: list[dict],
    ) -> None:
        """Run a local candidate audit synchronously through the compatibility interface."""

        self.ready_issues.extend(
            self.audit_candidate(
                case_id=case_id,
                base_step=base_step,
                mode="local",
                belief_before=belief_before,
                candidate_belief=belief_after,
                raw_trajectory=recent_trajectory,
            )
        )

    def submit_global(
        self,
        case_id: str,
        base_step: int,
        belief: dict,
        raw_trajectory: list[dict],
    ) -> None:
        """Run a global candidate audit synchronously through the compatibility interface."""

        self.ready_issues.extend(
            self.audit_candidate(
                case_id=case_id,
                base_step=base_step,
                mode="global",
                belief_before=belief,
                candidate_belief=belief,
                raw_trajectory=raw_trajectory,
            )
        )

    def submit_progress(
        self,
        *,
        case_id: str,
        step: int,
        frontier_episode: int,
        frontier: dict | None,
        action: Any,
        observation: str,
        task_mode: Literal["execution", "diagnostic"],
        belief_before: dict,
        belief_after: dict,
        diagnostic_threshold: float,
        frontier_history: list[dict[str, Any]] | None = None,
        done: bool = False,
        success: bool = False,
        task_score: float = 0.0,
    ) -> None:
        """Score a finalized transition only after the consistency stage."""

        if not self.progress_enabled:
            return
        if task_mode == "diagnostic":
            payload = diagnostic_progress(
                belief_before,
                belief_after,
                frontier,
                diagnostic_threshold,
            )
            method = "diagnostic_confidence_change"
        elif success:
            payload = {
                "progress": 1,
                "score": 1.0,
                "affected_record_ids": [],
                "reason": "The environment authoritatively reports task success.",
            }
            method = "execution_task_success"
        else:
            progress_input = {
                "step": step,
                "frontier": frontier,
                "action": action,
                "observation": observation,
                "belief_before": belief_before,
                "belief_after": belief_after,
            }
            payload = self._request_json(
                build_execution_progress_prompt(progress_input),
                self.call_progress_model,
                self.progress_max_json_retries,
                case_id=case_id,
                step=step,
                lane="progress",
                repair=build_progress_repair_prompt,
            )
            method = "execution_llm"
            if payload is None:
                self.logger.log(
                    "belief_sentinel_progress_failed",
                    case_id=case_id,
                    step=step,
                    action=action,
                    frontier=frontier,
                    error="invalid model response",
                )
                return
        try:
            self._validate_progress_payload(payload, belief_before, belief_after)
        except (TypeError, ValueError, KeyError) as error:
            self.logger.log(
                "belief_sentinel_progress_failed",
                case_id=case_id,
                step=step,
                action=action,
                frontier=frontier,
                error=str(error),
            )
            return
        finding = ProgressFinding(
            step=step,
            frontier_episode=frontier_episode,
            frontier=frontier,
            action=action,
            observation=observation,
            done=done,
            success=success,
            task_score=task_score,
            progress=int(payload["progress"]),
            scoring_method=method,
            score=float(payload["score"]),
            affected_record_ids=tuple(payload["affected_record_ids"]),
            reason=str(payload["reason"]),
        )
        self.ready_progress.append(finding)
        self.progress_history.append(asdict(finding))
        self.logger.log(
            "belief_sentinel_progress_completed",
            case_id=case_id,
            step=step,
            frontier_episode=frontier_episode,
            frontier=frontier,
            action=action,
            observation=observation,
            done=done,
            success=success,
            task_score=task_score,
            progress=finding.progress,
            scoring_method=method,
            score=finding.score,
            affected_record_ids=list(finding.affected_record_ids),
            reason=finding.reason,
        )

    def collect_ready(self) -> list[SentinelFinding]:
        """Drain completed consistency findings."""

        findings, self.ready_issues = self.ready_issues, []
        return findings

    def collect_progress_ready(self) -> list[ProgressFinding]:
        """Drain completed progress findings."""

        findings, self.ready_progress = self.ready_progress, []
        return findings

    def finish(self) -> list[SentinelFinding]:
        """Finish the episode; no background futures cross step boundaries."""

        self.closed = True
        return self.collect_ready()

    def _request_json(
        self,
        messages: list[dict[str, str]],
        caller: Callable[[list[dict]], str] | None,
        retries: int,
        *,
        case_id: str,
        step: int,
        lane: str,
        repair: Callable[[list[dict[str, str]], str], list[dict[str, str]]],
        audit_mode: str | None = None,
    ) -> dict | None:
        """Log model input and output and apply the configured bounded JSON repair."""

        if caller is None:
            return None
        event_prefix = (
            "belief_sentinel_model"
            if lane == "consistency"
            else "belief_sentinel_progress_model"
        )
        for attempt in range(retries + 1):
            fields = {
                "case_id": case_id,
                "step": step,
                "attempt": attempt,
                "messages": messages,
            }
            if audit_mode is not None:
                fields["audit_mode"] = audit_mode
            self.logger.log(f"{event_prefix}_input", **fields)
            raw = caller(messages)
            response_fields = dict(fields)
            response_fields.pop("messages")
            response_fields["raw_response"] = raw
            self.logger.log(f"{event_prefix}_response", **response_fields)
            try:
                return json.loads(raw)
            except json.JSONDecodeError as error:
                self.logger.log(
                    f"{event_prefix}_parse_error",
                    **response_fields,
                    error=str(error),
                )
                if attempt == retries:
                    return None
                messages = repair(messages, raw)
        return None

    def _compact_trajectory(self, trajectory: list[dict]) -> list[dict]:
        """Keep actions and bounded observations within the audit evidence budget."""

        compact = []
        for item in trajectory:
            observation = str(item.get("observation", ""))
            if len(observation) > self.audit_observation_max_chars:
                half = self.audit_observation_max_chars // 2
                observation = (
                    observation[:half] + "\n...[truncated]...\n" + observation[-half:]
                )
            compact.append(
                {
                    "step": item.get("step"),
                    "action": item.get("action"),
                    "observation": observation,
                }
            )
        return compact

    @staticmethod
    def _trajectory_text(item: dict) -> str:
        """Join an action and its observation into one evidence string."""

        return json.dumps(
            {"action": item.get("action"), "observation": item.get("observation")},
            ensure_ascii=False,
        )

    @staticmethod
    def _valid_targets(belief: dict) -> set[str]:
        """List belief record IDs and gap targets that issues may reference."""

        world = belief.get("world", {})
        targets = set()
        for kind in ("entities", "states", "relations"):
            targets.update(str(item) for item in world.get(kind, {}))
        for key in ("epistemic_gaps", "achievement_gaps"):
            targets.update(
                str(gap.get("target"))
                for gap in belief.get(key, [])
                if gap.get("target")
            )
        return targets

    @staticmethod
    def _valid_issue(
        issue: SentinelIssue,
        valid_targets: set[str],
        evidence_by_step: dict[int, str],
    ) -> bool:
        """Require a valid belief target and verbatim evidence from the cited step."""

        if not issue.targets or not set(issue.targets) <= valid_targets:
            return False
        evidence = evidence_by_step.get(issue.evidence_step)
        quote = " ".join(issue.evidence_quote.casefold().split())
        normalized = " ".join(str(evidence).casefold().split())
        return bool(quote and evidence is not None and quote in normalized)

    @staticmethod
    def _changed_record_ids(before: dict, after: dict) -> dict[str, list[str]]:
        """Collect IDs of added, removed, or modified world records."""

        result = {}
        for kind in ("entities", "states", "relations"):
            old = before.get(kind, {})
            new = after.get(kind, {})
            result[kind] = sorted(
                record_id
                for record_id in set(old) | set(new)
                if old.get(record_id) != new.get(record_id)
            )
        return result

    @classmethod
    def _validate_progress_payload(
        cls,
        payload: dict,
        before: dict,
        after: dict,
    ) -> None:
        """Validate the binary progress label and referenced affected record IDs."""

        if payload["progress"] not in (0, 1) or isinstance(payload["progress"], bool):
            raise ValueError("progress must be integer 0 or 1")
        score = float(payload["score"])
        if score < 0.0:
            raise ValueError("score must be non-negative")
        if not isinstance(payload["affected_record_ids"], list):
            raise TypeError("affected_record_ids must be a list")
        if not isinstance(payload["reason"], str):
            raise TypeError("reason must be a string")
        changed = set().union(
            *cls._changed_record_ids(
                before.get("world", {}),
                after.get("world", {}),
            ).values()
        )
        unknown = set(payload["affected_record_ids"]) - changed
        if unknown:
            raise ValueError(f"affected_record_ids did not change: {sorted(unknown)}")
