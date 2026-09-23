"""Uncompressed, episode-local action-observation history."""

from __future__ import annotations


class RawTrajectoryContext:
    """Keep the complete uncompressed interaction history within one episode."""

    def __init__(self):
        """Initialize configuration and episode-local runtime state."""

        self.trajectory: list[dict] = []

    def reset(self, state: dict) -> None:
        """Clear the previous episode history."""

        self.trajectory = []

    def update(
        self,
        transition: dict,
        state: dict,
        step: int,
    ) -> None:
        """Append the action-observation pair in execution order."""

        self.trajectory.append(transition)

    def get_context(self) -> dict:
        """Return the uncompressed trajectory for the Raw policy prompt."""

        return {
            "type": "raw",
            "trajectory": self.trajectory,
        }

    def get_action_override(self, state: dict) -> object | None:
        """Return no direct action override; actions are chosen by the task policy."""

        return None

    def get_result(self) -> dict:
        """Return no duplicate trajectory; the policy loop already exports it."""

        return {}
