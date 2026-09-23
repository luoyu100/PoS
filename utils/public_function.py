"""Transient request retries and readable event-trace rendering."""

from __future__ import annotations
import json
import random
import sys
from pathlib import Path
from time import sleep
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

TRANSIENT_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)
TRANSIENT_MAX_ATTEMPTS = 8
TRANSIENT_BASE_DELAY = 2.0
TRANSIENT_MAX_DELAY = 60.0


def call_with_backoff(operation, description: str):
    """Retry configured transient failures with the existing bounded backoff policy."""
    for attempt in range(1, TRANSIENT_MAX_ATTEMPTS + 1):
        try:
            return operation()
        except TRANSIENT_ERRORS as error:
            if attempt == TRANSIENT_MAX_ATTEMPTS:
                raise
            delay = min(TRANSIENT_BASE_DELAY * 2 ** (attempt - 1), TRANSIENT_MAX_DELAY)
            delay += random.uniform(0, delay / 2)
            print(
                f"{description}: {type(error).__name__}，{delay:.1f}s before retry (attempt {attempt}/{TRANSIENT_MAX_ATTEMPTS - 1})",
                file=sys.stderr,
                flush=True,
            )
            sleep(delay)


ROLE_TITLES = {"system": "⚙️ System", "user": "🧑 User", "assistant": "🤖 Assistant"}


def _render_messages(messages: list[dict], heading: str) -> list[str]:
    """Render messages."""
    lines = []
    for message in messages:
        title = ROLE_TITLES.get(message["role"], message["role"].title())
        lines.extend(
            [f"{heading} {title}", "", "```text", message["content"], "```", ""]
        )
    return lines


def _load_events(events_path: Path) -> list[dict]:
    """Load events."""
    return [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]


def _render_event_steps(events: list[dict], level: int) -> list[str]:
    """Render event steps."""
    step_heading = "#" * level
    message_heading = "#" * (level + 1)
    lines = []
    for event in events:
        if event["event"] == "react_model_input":
            attempt = event.get("attempt", 0)
            retry = f" · Retry {attempt}" if attempt else ""
            lines.extend(
                [
                    "---",
                    "",
                    f"{step_heading} Step {event['step'] + 1}{retry}",
                    "",
                    f"> Input timestamp: `{event['time']}`",
                    "",
                ]
            )
            lines.extend(_render_messages(event["messages"], message_heading))
        elif event["event"] == "react_model_response":
            attempt = event.get("attempt", 0)
            retry = f" · Retry {attempt}" if attempt else ""
            lines.extend(
                [
                    f"{message_heading} 🤖 Raw response{retry}",
                    "",
                    "```json",
                    event["raw_response"],
                    "```",
                    "",
                ]
            )
        elif event["event"] == "react_model_parse_error":
            lines.extend([f"> JSON parse failed: `{event['error']}`", ""])
        elif event["event"] == "terminal_context_error":
            lines.extend(
                [
                    f"{message_heading} Terminal context error",
                    "",
                    "The environment result is preserved; no further action is taken.",
                    "",
                    "~~~json",
                    json.dumps(
                        {
                            key: event[key]
                            for key in ("step", "phase", "error_type", "error")
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "~~~",
                    "",
                ]
            )
        elif event["event"] == "belief_model_input":
            operation = event["operation"]
            if operation == "initialize":
                title = "🌐 Initial belief construction"
            elif operation == "update":
                title = f"🌐 Belief update after Step {event['step'] + 1}"
            elif event["step"] == -1:
                title = "📝 Initial belief text generation"
            else:
                title = f"📝 Belief text generation after Step {event['step'] + 1}"
            attempt = event.get("attempt", 0)
            retry = f" · Retry {attempt}" if attempt else ""
            lines.extend(
                [
                    "---",
                    "",
                    f"{step_heading} {title}{retry}",
                    "",
                    f"> Input timestamp: `{event['time']}`",
                    "",
                    *_render_messages(event["messages"], message_heading),
                ]
            )
        elif event["event"] == "belief_model_response":
            attempt = event.get("attempt", 0)
            retry = f" · Retry {attempt}" if attempt else ""
            response_name = (
                "Belief text model response"
                if event["operation"] == "textualize"
                else "Belief model response"
            )
            lines.extend(
                [
                    f"{message_heading} 🤖 {response_name}{retry}",
                    "",
                    "```json",
                    event["raw_response"],
                    "```",
                    "",
                ]
            )
        elif event["event"] == "belief_model_parse_error":
            lines.extend([f"> Belief JSON parse failed: `{event['error']}`", ""])
        elif event["event"] == "belief_sentinel_audit_submitted":
            audit_name = (
                "Local audit" if event["audit_mode"] == "local" else "Global audit"
            )
            lines.extend(
                [
                    "---",
                    "",
                    f"{step_heading} 🛡️ BeliefSentinel {audit_name} after Step {event['step'] + 1}",
                    "",
                    f"> Submission timestamp: `{event['time']}`",
                    "",
                ]
            )
        elif event["event"] == "belief_sentinel_model_input":
            attempt = event.get("attempt", 0)
            if attempt:
                lines.extend([f"> Retry `{attempt}`", ""])
            lines.extend(_render_messages(event["messages"], message_heading))
        elif event["event"] == "belief_sentinel_model_response":
            attempt = event.get("attempt", 0)
            retry = f" · Retry {attempt}" if attempt else ""
            lines.extend(
                [
                    f"{message_heading} 🐞 Sentinel response{retry}",
                    "",
                    "```json",
                    event["raw_response"],
                    "```",
                    "",
                ]
            )
        elif event["event"] == "belief_sentinel_model_parse_error":
            lines.extend(
                [f"> BeliefSentinel JSON parse failed: `{event['error']}`", ""]
            )
        elif event["event"] == "belief_sentinel_audit_completed":
            lines.extend(
                [
                    f"> BeliefSentinel audit completed with `{event['issue_count']}` issues",
                    "",
                    "```json",
                    json.dumps(event["issues"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
        elif event["event"] == "belief_sentinel_audit_failed":
            lines.extend(
                [
                    f"> BeliefSentinel audit failed without aborting the agent: {event.get('error_type', 'audit_error')}",
                    "",
                    "```text",
                    event["error"],
                    "```",
                    "",
                ]
            )
        elif event["event"] == "belief_sentinel_progress_submitted":
            lines.extend(
                [
                    "---",
                    "",
                    f"{step_heading} 📈 Progress scoring for original Step {event['step'] + 1}",
                    "",
                    f"> Action：{json.dumps(event['action'], ensure_ascii=False)}  ",
                    f"> Frontier episode：{event['frontier_episode']}  ",
                    f"> Outcome：done={event.get('done')} · success={event.get('success')} · score={event.get('task_score')}  ",
                    f"> Frontier：{json.dumps(event.get('frontier'), ensure_ascii=False)}",
                    "",
                ]
            )
        elif event["event"] == "belief_sentinel_progress_model_input":
            attempt = event.get("attempt", 0)
            if attempt:
                lines.extend([f"> Progress retry {attempt}", ""])
            lines.extend(_render_messages(event["messages"], message_heading))
        elif event["event"] == "belief_sentinel_progress_model_response":
            attempt = event.get("attempt", 0)
            retry = f" · Retry {attempt}" if attempt else ""
            lines.extend(
                [
                    f"{message_heading} 🐞 Progress response{retry}",
                    "",
                    "~~~json",
                    event["raw_response"],
                    "~~~",
                    "",
                ]
            )
        elif event["event"] == "belief_sentinel_progress_model_parse_error":
            lines.extend(
                [f"> Progress JSON parsing or validation failed: {event['error']}", ""]
            )
        elif event["event"] == "belief_sentinel_progress_completed":
            lines.extend(
                [
                    f"{message_heading} ✅ Progress for original Step {event['step'] + 1}",
                    "",
                    f"> Action：{json.dumps(event['action'], ensure_ascii=False)}  ",
                    f"> u_t：{event['progress']}  ",
                    f"> Method：{event['scoring_method']}  ",
                    f"> Score：{event['score']}  ",
                    f"> Reason：{event['reason']}",
                    "",
                ]
            )
        elif event["event"] == "belief_sentinel_progress_failed":
            lines.extend(
                [
                    f"> ⚠️ Original Step {event['step'] + 1} remains unverified; it is not counted as zero progress in stagnation",
                    "",
                    "~~~text",
                    event["error"],
                    "~~~",
                    "",
                ]
            )
        elif event["event"] == "belief_sentinel_issues_injected":
            lines.extend(
                [
                    f"> Pass `{event['issue_count']}` pending issues to BeliefManager",
                    "",
                    "```json",
                    json.dumps(event["issues"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
        elif event["event"] == "belief_sentinel_issues_resolved":
            lines.extend(
                [
                    f"{message_heading} ✅ BeliefManager issue decisions",
                    "",
                    "```json",
                    json.dumps(event["resolutions"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
        elif event["event"] == "belief_candidate_generated":
            lines.extend(
                [
                    f"{message_heading} Candidate Belief (uncommitted)",
                    "",
                    "~~~json",
                    json.dumps(event["candidate_belief"], ensure_ascii=False, indent=2),
                    "~~~",
                    "",
                ]
            )
        elif event["event"] == "belief_candidate_rejected":
            lines.extend(
                [
                    "> Candidate rejected by Sentinel; retain the previous B_t.",
                    "",
                    "~~~json",
                    json.dumps(
                        event.get("issues", event.get("error")),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "~~~",
                    "",
                ]
            )
        elif event["event"] == "belief_candidate_repaired":
            lines.extend(["> Candidate validated after one issue-driven repair.", ""])
        elif event["event"] == "belief_text_updated":
            lines.extend(
                [
                    f"{message_heading} 📝 Derived belief text",
                    "",
                    "```text",
                    event["belief_text"],
                    "```",
                    "",
                ]
            )
        elif event["event"] == "belief_frontier_updated":
            lines.extend(
                [
                    f"{message_heading} 🧭 Frontier episode {event['frontier_episode']}",
                    "",
                    "~~~json",
                    json.dumps(
                        {
                            "previous": event.get("previous"),
                            "current": event.get("current"),
                            "current_gap": event.get("current_gap"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "~~~",
                    "",
                ]
            )
        elif event["event"] == "belief_health_updated":
            health = event["health"]
            lines.extend(
                [
                    f"{message_heading} 🩺 Belief Health",
                    "",
                    f"> Ready：{health['window_ready']} · P_E/P_A：{health['persistence_e']:.3f} / {health['persistence_a']:.3f} · S：{health['stagnation']:.3f}  ",
                    f"> R: {health['recurrence']:.3f} · H: {health['health_score']:.3f} · Health threshold: {health['health_threshold']:.3f}  ",
                    f"> Type：{health.get('gap_dimension')} · Pattern：{health.get('trapping_pattern')}",
                    "",
                ]
            )
        elif event["event"] == "belief_recovery_started":
            lines.extend(
                [
                    f"{message_heading} 🚨 Belief Trapping / Recovery started",
                    "",
                    "~~~json",
                    json.dumps(event["recovery"], ensure_ascii=False, indent=2),
                    "~~~",
                    "",
                ]
            )
        elif event["event"] == "belief_recovery_finished":
            lines.extend(
                [
                    f"{message_heading} 🛟 Recovery {event['status']}",
                    "",
                    "~~~json",
                    json.dumps(event["recovery"], ensure_ascii=False, indent=2),
                    "~~~",
                    "",
                ]
            )
        elif event["event"] in {"belief_state_initialized", "belief_state_updated"}:
            lines.extend(
                [
                    f"> Belief state updated: `{event['entities']}` entities，`{event['states']}` states，`{event['relations']}` relations，`{event['epistemic_gaps']}` epistemic gaps，`{event['achievement_gaps']}` achievement gaps",
                    "",
                    "~~~json",
                    json.dumps(event["belief_state"], ensure_ascii=False, indent=2),
                    "~~~",
                    "",
                ]
            )
    return lines


def render_events_markdown(events_path: Path, markdown_path: Path) -> None:
    """Render an append-only event stream as a readable Markdown trace."""
    events = _load_events(events_path)
    lines = [
        "# ReAct Message Trace",
        "",
        "## 🧩 Case",
        "",
        f"`{events[0]['case_id']}`",
        "",
        f"> Raw events: `{events_path.name}`",
        "",
        *_render_event_steps(events, level=3),
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def render_case_index_markdown(case_paths: list[Path], markdown_path: Path) -> None:
    """Render a collapsible index of completed case traces."""
    lines = [
        "# ReAct Case Browser",
        "",
        "> Expand a case below to inspect its complete message trace.",
        "",
    ]
    for case_path in case_paths:
        result = json.loads((case_path / "result.json").read_text(encoding="utf-8"))
        events = _load_events(case_path / "events.jsonl")
        status = "Success" if result["success"] else "Incomplete"
        relative_path = case_path.relative_to(markdown_path.parent)
        lines.extend(
            [
                "<details>",
                f"<summary><strong>{case_path.name}</strong> · {status} · {result['steps']} steps</summary>",
                "",
                f"> **Case ID**：`{result['case_id']}`  ",
                f"> **Task**：{result['task']}  ",
                f"> [Open Markdown]({relative_path}/events.md) | [Raw JSONL]({relative_path}/events.jsonl)",
                "",
                *_render_event_steps(events, level=3),
                "</details>",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
