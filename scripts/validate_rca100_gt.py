"""Validate rca100 gt utilities."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.rca100_env import normalize_entity, normalize_fault_type

DATA = ROOT / "benchmarks" / "RCA100"
ALLOWED_LEVELS = {"apm.service", "k8s.node"}


def entity_types(topology: dict, name: str) -> set[str]:
    """Entity types."""

    normalized = normalize_entity(name)
    return {
        entity["type"]
        for entity in topology["entities"]
        if normalize_entity(entity["name"]) == normalized
    }


def main() -> int:
    """Main."""

    manifest = [
        line.strip()
        for line in (DATA / "manifest.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    violations: list[str] = []
    level_counter: Counter = Counter()

    for task_id in manifest:
        ground_truth = json.loads(
            (DATA / "answer_key" / f"{task_id}.gt.json").read_text(encoding="utf-8")
        )
        raw = json.loads(ground_truth["raw_ground_truth"])
        entities = ground_truth["root_cause_entities"]

        if len(entities) != 1:
            violations.append(
                f"{task_id}: root_cause_entities has {len(entities)} items"
            )

        topology = json.loads(
            (DATA / "cases" / task_id / "topology.json").read_text(encoding="utf-8")
        )
        for name in entities:
            types = entity_types(topology, name)
            if not types:
                violations.append(
                    f"{task_id}: entity {name!r} is not present in its "
                    "own topology.json"
                )
                level_counter["UNRESOLVED"] += 1
                continue
            if not types & ALLOWED_LEVELS:
                violations.append(
                    f"{task_id}: entity {name!r} resolves to {sorted(types)}, "
                    "not service/node level"
                )
            level_counter["|".join(sorted(types))] += 1

        for value in ground_truth.get("root_cause_types", []):
            normalized = normalize_fault_type(value)
            if normalized is None:
                violations.append(
                    f"{task_id}: root_cause_type {value!r} is outside "
                    "the 29-type closed set"
                )
            else:
                expected = normalize_fault_type(raw["outcome"]["expected_fault_id"])
                if normalized != expected:
                    violations.append(
                        f"{task_id}: root_cause_type {value!r} != "
                        f"expected_fault_id {raw['outcome']['expected_fault_id']!r}"
                    )

        target_names = {
            normalize_entity(target["entity_name"])
            for target in raw["outcome"].get("target_entities", [])
        }
        gt_names = {normalize_entity(name) for name in entities}
        if target_names and target_names != gt_names:
            violations.append(
                f"{task_id}: root_cause_entities {sorted(gt_names)} differ "
                f"from raw_ground_truth target_entities {sorted(target_names)}"
            )

    print(f"cases checked: {len(manifest)}")
    print("entity level distribution (via per-case topology):")
    for key, count in level_counter.most_common():
        print(f"  {count:4d}  {key}")
    print(f"violations: {len(violations)}")
    for violation in violations:
        print(f"  - {violation}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
