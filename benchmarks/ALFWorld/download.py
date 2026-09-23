"""Prepare ALFWorld data using the shared, non-overwriting download utility."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from prepare_benchmarks import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--benchmark", "alfworld"]
    main()
