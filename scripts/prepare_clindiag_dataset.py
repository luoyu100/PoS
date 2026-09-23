"""Reproduce the fixed ClinDiag subset from an authorized upstream archive."""

from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "ClinDiag"
ENVIRONMENT_FILES = (
    "initial_information.json",
    "medical_history.json",
    "physical_examination.json",
    "diagnostic_test.json",
)
SAMPLE_SEED = 42
SAMPLE_SIZE = 302


def case_ids(archive: zipfile.ZipFile) -> tuple[list[str], list[str]]:
    """Case ids."""

    tops = sorted({name.split("/")[0] for name in archive.namelist()})
    rare = [top for top in tops if top.startswith("rare")]
    challenging = [top for top in tops if not top.startswith("rare")]
    return rare, challenging


def diagnosis_valid(archive: zipfile.ZipFile, case_id: str) -> bool:
    """Diagnosis valid."""

    try:
        payload = json.loads(archive.read(f"{case_id}/diagnosis.json").decode("utf-8"))
    except (KeyError, ValueError):
        return False
    final = str(payload.get("diagnosis", {}).get("final_diagnosis", "")).strip()
    return bool(final) and final.lower() not in {"not specified", "none"}


def build_dataset(source: Path, output: Path) -> None:
    """Build into a new staging directory; the caller verifies before installing."""
    if output.exists():
        raise ValueError("The staging directory must not already exist")
    with zipfile.ZipFile(source) as archive:
        _write_dataset(archive, output)


def _write_dataset(archive: zipfile.ZipFile, output: Path) -> None:
    """Keep valid rare cases and deterministically sample challenging cases."""
    rare, challenging = case_ids(archive)
    print(f"zip contains rare={len(rare)} challenging={len(challenging)}")

    rare_selected = [case for case in rare if diagnosis_valid(archive, case)]
    challenging_pool = [case for case in challenging if diagnosis_valid(archive, case)]
    excluded = sorted(
        set(rare + challenging) - set(rare_selected) - set(challenging_pool)
    )
    print(f"excluded (unusable diagnosis.json): {excluded}")

    challenging_selected = sorted(
        random.Random(SAMPLE_SEED).sample(challenging_pool, SAMPLE_SIZE)
    )

    cases_dir = output / "cases"
    answer_dir = output / "answer_key"
    cases_dir.mkdir(parents=True, exist_ok=True)
    answer_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[str] = []
    for subset, selected in (
        ("rare", rare_selected),
        ("challenging", challenging_selected),
    ):
        for case_id in selected:
            if case_id in {".", ".."} or "/" in case_id or "\\" in case_id:
                raise ValueError(f"Unsafe case identifier: {case_id}")
            slug = f"{subset}__{case_id}"
            case_dir = cases_dir / slug
            case_dir.mkdir(parents=True, exist_ok=True)
            for filename in ENVIRONMENT_FILES:
                source = f"{case_id}/{filename}"
                try:
                    payload = archive.read(source)
                except KeyError:
                    continue
                (case_dir / filename).write_bytes(payload)
            (answer_dir / f"{slug}.diagnosis.json").write_bytes(
                archive.read(f"{case_id}/diagnosis.json")
            )
            manifest.append(f"{subset}/{case_id}")

    (output / "manifest.txt").write_text(
        "\n".join(manifest) + "\n",
        encoding="utf-8",
    )

    card = f"""# ClinDiag for PoS — Dataset Card

- Source: ClinDiag-Benchmark (`Clindiag_Benchmark.zip`),
  https://github.com/geteff1/ClinDiag (paper: Nature Communications,
  s41467-026-70274-w). Please follow the upstream license and cite the
  original paper when redistributing.
- Built on: {date.today().isoformat()}
- Subsets: Rare Disease Subset kept in full ({len(rare_selected)} cases);
  Challenging Case Subset deterministically sampled
  ({len(challenging_selected)} of {len(challenging_pool)} usable cases,
  `random.Random({SAMPLE_SEED}).sample` over sorted case ids).
- Excluded upstream cases with unusable `diagnosis.json`: {excluded}.
- Layout: `cases/<subset>__<case_id>/` holds only the four
  environment-side files ({", ".join(ENVIRONMENT_FILES)});
  `answer_key/<subset>__<case_id>.diagnosis.json` holds the ground truth
  and must never enter any agent-facing prompt.
- Case order for experiments is fixed by `manifest.txt`
  (rare first, then challenging; each group sorted by case id).
"""
    (output / "DATASET_CARD.md").write_text(card, encoding="utf-8")

    print(
        f"done: {len(manifest)} cases -> {cases_dir} "
        f"(rare={len(rare_selected)}, challenging={len(challenging_selected)})"
    )


def main() -> int:
    """Keep the legacy CLI while using the non-overwriting preparation entry."""
    import subprocess

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, help="Official Clindiag_Benchmark.zip")
    args = parser.parse_args()
    return subprocess.call(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_benchmarks.py"),
            "--benchmark",
            "clindiag",
            "--source",
            args.zip,
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
