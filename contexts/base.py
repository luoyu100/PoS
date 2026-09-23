"""Interface shared by decision-context providers."""

from __future__ import annotations

from typing import Protocol


class ContextProvider(Protocol):
    """Episode-local context consumed by the shared policy loop."""

    def reset(self, state: dict) -> None:
        """Initialize context from the benchmark's first observation."""

        ...

    def update(
        self,
        transition: dict,
        state: dict,
        step: int,
    ) -> None:
        """Consume one completed action–observation transition."""

        ...

    def get_context(self) -> dict:
        """Return the next decision's context without changing episode state."""

        ...

    def get_result(self) -> dict:
        """Export method-specific state for the episode result and trace."""

        ...
