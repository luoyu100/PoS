"""Download benchmark inputs or import a local copy without overwriting data."""

from __future__ import annotations

import argparse
import filecmp
import json
from pathlib import Path
import shutil
import stat
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "alfworld": "ALFWorld",
    "loca": "LOCA-Bench",
    "rca100": "RCA100",
    "clindiag": "ClinDiag",
}
RCA_URL = "https://aiops-benchmark.oss-cn-hongkong.aliyuncs.com/rca/rca100/v1.1/"
RCA_FILES = (
    "task.json",
    "topology.json",
    "alerts.parquet",
    "events.parquet",
    "logs.parquet",
    "metrics.parquet",
    "traces.parquet",
)
LENGTHS = ("8k", "16k", "32k", "64k", "96k", "128k", "256k")


def download(url: str, target: Path) -> None:
    """Reuse completed downloads and atomically publish new files."""
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "PoS-data-preparation"}
        )
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            partial.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def copy_data(source: Path, target: Path) -> None:
    """Check every conflict before copying missing files; never replace local edits."""
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"No data files found in {source}")
    for path in files:
        destination = target / path.relative_to(source)
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"Source must contain regular files: {path}")
        if destination.is_symlink() or not destination.resolve().is_relative_to(
            target.resolve()
        ):
            raise ValueError(f"Unsafe destination: {destination}")
        if destination.exists() and (
            not destination.is_file()
            or not filecmp.cmp(path, destination, shallow=False)
        ):
            raise ValueError(f"Refusing to overwrite different data: {destination}")
    for path in files:
        destination = target / path.relative_to(source)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)


def prepare_alfworld(target: Path, source: Path | None, work: Path) -> None:
    """Install the valid-unseen text games and matching package-provided logic."""
    if source is None:
        source = work / "alfworld"
        for suffix in (
            "0.2.2/json_2.1.1_json.zip",
            "0.2.2/json_2.1.1_pddl.zip",
            "0.4.0/json_2.1.2_tw-pddl.zip",
        ):
            archive = work / Path(suffix).name
            print(f"Downloading {archive.name}", flush=True)
            download(
                "https://github.com/alfworld/alfworld/releases/download/" + suffix,
                archive,
            )
            with zipfile.ZipFile(archive) as data:
                for entry in data.infolist():
                    relative = Path(entry.filename)
                    if (
                        relative.is_absolute()
                        or ".." in relative.parts
                        or stat.S_ISLNK(entry.external_attr >> 16)
                    ):
                        raise ValueError(f"Unsafe archive entry: {entry.filename}")
                    if entry.is_dir() or not entry.filename.startswith(
                        "json_2.1.1/valid_unseen/"
                    ):
                        continue
                    destination = source / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        data.open(entry) as incoming,
                        destination.open("wb") as outgoing,
                    ):
                        shutil.copyfileobj(incoming, outgoing)
        from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH

        (source / "logic").mkdir(parents=True)
        shutil.copyfile(ALFRED_PDDL_PATH, source / "logic/alfred.pddl")
        shutil.copyfile(ALFRED_TWL2_PATH, source / "logic/alfred.twl2")
    copy_data(source / "json_2.1.1/valid_unseen", target / "json_2.1.1/valid_unseen")
    copy_data(source / "logic", target / "logic")


def prepare_rca100(target: Path, source: Path | None, answer_key: Path | None) -> None:
    """Download observations while keeping evaluator labels separate."""
    if answer_key:
        copy_data(answer_key, target / "answer_key")
    case_ids = (target / "manifest.txt").read_text().split()
    for case_id in case_ids:
        if not (target / "answer_key" / f"{case_id}.gt.json").is_file():
            raise ValueError(f"Missing evaluator label: {case_id}")
    if source:
        copy_data(source / "cases", target / "cases")
        return
    for case_id in case_ids:
        print(f"Preparing RCA100 {case_id}", flush=True)
        for name in RCA_FILES:
            relative = f"cases/{case_id}/{name}"
            download(RCA_URL + relative, target / relative)


def prepare_clindiag(target: Path, source: Path | None, work: Path) -> None:
    """Reproduce the fixed 604-case ordering from an authorized archive."""
    if source is None:
        raise ValueError(
            "Supply --source /path/to/Clindiag_Benchmark.zip from https://github.com/geteff1/ClinDiag"
        )
    if source.is_file():
        from prepare_clindiag_dataset import build_dataset

        stage = work / "clindiag"
        build_dataset(source, stage)
        source = stage
    if (source / "manifest.txt").read_text().split() != (
        target / "manifest.txt"
    ).read_text().split():
        raise ValueError("ClinDiag case order does not match the fixed 604-case subset")
    copy_data(source / "cases", target / "cases")
    copy_data(source / "answer_key", target / "answer_key")


def check_data(benchmark: str, root: Path) -> None:
    """Check the required data layout without a separate checksum database."""
    if benchmark == "loca":
        for length in LENGTHS:
            config = json.loads(
                (root / f"task-configs/final_{length}_set_config.json").read_text()
            )
            if len(config["configurations"]) != 75:
                raise ValueError(f"Expected 75 LOCA tasks at {length}")
        return
    if benchmark == "alfworld":
        games = list((root / "json_2.1.1/valid_unseen").rglob("game.tw-pddl"))
        if len(games) != 134:
            raise ValueError(f"Expected 134 ALFWorld games, found {len(games)}")
        required = [root / "logic/alfred.pddl", root / "logic/alfred.twl2"]
    elif benchmark == "rca100":
        case_ids = (root / "manifest.txt").read_text().split()
        if len(case_ids) != 103:
            raise ValueError("Expected 103 RCA100 cases")
        required = [
            root / "cases" / case_id / name
            for case_id in case_ids
            for name in RCA_FILES
        ]
        required += [root / "answer_key" / f"{case_id}.gt.json" for case_id in case_ids]
    else:
        names = (
            "initial_information.json",
            "medical_history.json",
            "physical_examination.json",
        )
        slugs = [
            case.replace("/", "__", 1)
            for case in (root / "manifest.txt").read_text().split()
        ]
        # Three official cases omit diagnostic_test.json; the environment supports this.
        if len(slugs) != 604:
            raise ValueError("Expected 604 ClinDiag cases")
        required = [root / "cases" / slug / name for slug in slugs for name in names]
        required += [root / "answer_key" / f"{slug}.diagnosis.json" for slug in slugs]
    for path in required:
        if not path.is_file():
            raise ValueError(f"Missing required file: {path}")


def main() -> None:
    """Prepare data without starting tasks or making model calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=NAMES)
    parser.add_argument(
        "--source",
        type=Path,
        help="Existing dataset directory or official ClinDiag ZIP",
    )
    parser.add_argument(
        "--answer-key", type=Path, help="Optional RCA100 label directory"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check required files without downloading",
    )
    args = parser.parse_args()
    target = ROOT / "benchmarks" / NAMES[args.benchmark]
    try:
        if not args.source and not args.answer_key and not args.verify_only:
            try:
                check_data(args.benchmark, target)
            except (FileNotFoundError, ValueError):
                pass
            else:
                print(f"{NAMES[args.benchmark]} is already ready.")
                return
        if not args.verify_only and args.benchmark != "loca":
            with tempfile.TemporaryDirectory(prefix="pos-data-") as directory:
                work = Path(directory)
                if args.benchmark == "alfworld":
                    prepare_alfworld(target, args.source, work)
                elif args.benchmark == "rca100":
                    prepare_rca100(target, args.source, args.answer_key)
                else:
                    prepare_clindiag(target, args.source, work)
        check_data(args.benchmark, target)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")
    print(f"{NAMES[args.benchmark]} is ready.")


if __name__ == "__main__":
    main()
