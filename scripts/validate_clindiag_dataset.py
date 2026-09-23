"""Validate clindiag dataset utilities."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "ClinDiag"


def main() -> int:
    """Main."""

    manifest = [
        line.strip()
        for line in (DATA / "manifest.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    violations: list[str] = []
    subset_counter: Counter = Counter()

    disk_cases = {path.name for path in (DATA / "cases").iterdir()}
    for case_ref in manifest:
        subset, upstream_id = case_ref.split("/", maxsplit=1)
        subset_counter[subset] += 1
        slug = f"{subset}__{upstream_id}"

        if slug not in disk_cases:
            violations.append(f"{case_ref}: case directory missing")
            continue
        case_dir = DATA / "cases" / slug

        try:
            initial = json.loads(
                (case_dir / "initial_information.json").read_text(encoding="utf-8")
            )["initial_information"]
            if not str(initial).strip():
                violations.append(f"{case_ref}: empty initial_information")
        except (OSError, ValueError, KeyError) as error:
            violations.append(
                f"{case_ref}: unusable initial_information.json ({error})"
            )

        answer_path = DATA / "answer_key" / f"{slug}.diagnosis.json"
        try:
            final = json.loads(answer_path.read_text(encoding="utf-8"))["diagnosis"][
                "final_diagnosis"
            ]
            if not str(final).strip() or str(final).strip().lower() in {
                "not specified",
                "none",
            }:
                violations.append(f"{case_ref}: empty final_diagnosis")
        except (OSError, ValueError, KeyError) as error:
            violations.append(f"{case_ref}: unusable answer_key diagnosis ({error})")

        leftovers = [
            path.name
            for path in case_dir.iterdir()
            if path.name == "diagnosis.json" or path.name.endswith("_raw.txt")
        ]
        if leftovers:
            violations.append(
                f"{case_ref}: answer/leftover files inside case dir: {leftovers}"
            )

    extras = disk_cases - {
        f"{ref.split('/', 1)[0]}__{ref.split('/', 1)[1]}" for ref in manifest
    }
    if extras:
        violations.append(f"case dirs not in manifest: {sorted(extras)[:5]}")

    print(f"cases checked: {len(manifest)} | subsets: {dict(subset_counter)}")
    print(f"violations: {len(violations)}")
    for violation in violations:
        print(f"  - {violation}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
