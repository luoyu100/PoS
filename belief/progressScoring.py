"""Progress scoring from absolute inferred-confidence changes and observed outcomes."""

from __future__ import annotations

import re
import unicodedata


def diagnostic_progress(
    belief_before: dict,
    belief_after: dict,
    frontier: dict | None,
    threshold: float,
) -> dict:
    """Compute d_diag = 0.5 * sum(abs(c_new - c_old)); progress requires d_diag > epsilon."""

    changes = _inferred_confidence_changes(
        belief_before["world"],
        belief_after["world"],
    )
    distance = round(0.5 * sum(changes.values()), 12)
    return {
        "progress": int(distance > threshold),
        "score": distance,
        "affected_record_ids": sorted(
            {
                record_id.split(":", 1)[1]
                for record_id, change in changes.items()
                if change > 0.0
            }
        ),
        "reason": (
            f"Diagnostic confidence distance is {distance:.4f}; "
            f"progress requires distance > {threshold:.4f}."
        ),
    }


def _inferred_confidence_changes(
    before: dict,
    after: dict,
) -> dict[str, float]:
    """Align inferred records by stable ID; missing records have zero confidence."""

    changes: dict[str, float] = {}
    for kind in ("states", "relations"):
        before_records = before.get(kind, {})
        after_records = after.get(kind, {})
        for record_id in set(before_records) | set(after_records):
            old = before_records.get(record_id)
            new = after_records.get(record_id)
            old_inferred = old is not None and old.get("source_type") == "inferred"
            new_inferred = new is not None and new.get("source_type") == "inferred"
            if not (old_inferred or new_inferred):
                continue

            old_confidence = float(old.get("probability", 0.0)) if old_inferred else 0.0
            new_confidence = float(new.get("probability", 0.0)) if new_inferred else 0.0
            changes[f"{kind}:{record_id}"] = abs(new_confidence - old_confidence)
    return changes


def removed_gap_targets(before: list[dict], after: list[dict]) -> set[str]:
    """Find resolved gaps without counting semantically equivalent rewordings."""

    remaining = [str(gap.get("target", "")) for gap in after]
    return {
        _normalize(str(gap.get("target", "")))
        for gap in before
        if not any(
            _same_gap_semantics(str(gap.get("target", "")), candidate)
            for candidate in remaining
        )
    }


_VOLATILE = re.compile(
    r"(?:\b[0-9a-f]{8}-[0-9a-f-]{27,}\b|\b\d+\b|0x[0-9a-f]+)",
    flags=re.IGNORECASE,
)


def progress_outcome_signature(observation: object) -> str:
    """Normalize a failed outcome while ignoring volatile IDs and numbers."""

    text = unicodedata.normalize("NFKC", str(observation)).casefold()
    return re.sub(r"\s+", " ", _VOLATILE.sub("<value>", text)).strip()[:600]


def _same_gap_semantics(left: str, right: str) -> bool:
    """Conservatively align gap meaning using normalized token overlap."""

    if _normalize(left) == _normalize(right):
        return True
    left_tokens = set(re.findall(r"\w+", _normalize(left), flags=re.UNICODE))
    right_tokens = set(re.findall(r"\w+", _normalize(right), flags=re.UNICODE))
    if min(len(left_tokens), len(right_tokens)) < 2:
        return False
    shared = left_tokens & right_tokens
    return (
        len(shared) / min(len(left_tokens), len(right_tokens)) >= 0.75
        and len(shared) / len(left_tokens | right_tokens) >= 0.5
    )


def _normalize(text: str) -> str:
    """Normalize Unicode, case, and whitespace."""

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())
