# Benchmark Setup

Run commands from the repository root with `conda activate PoS`.
Runtime integrations are included; large evaluation data and generated workspaces
are not. Existing files are reused and local imports refuse to overwrite
different data. File checks validate layout and counts, not a checksum database.

## ALFWorld

```bash
pip install -r requirements/alfworld.txt
python benchmarks/ALFWorld/download.py
```

This installs the 134-case valid-unseen text-game split under
`benchmarks/ALFWorld/json_2.1.1/valid_unseen/`, with matching files in `logic/`.
The unified adapter uses the official ALFWorld 0.4.2 Python environment.
For an existing dataset:

```bash
python benchmarks/ALFWorld/download.py --source /path/to/alfworld-data
```

The source directory should contain `json_2.1.1/valid_unseen/` and `logic/`.
Linux x86_64 is recommended. TextWorld source builds may require compilers and
upstream Inform7 downloads; its prebuilt Linux wheel avoids that source-build path.

## LOCA-Bench

The environment code and task configurations are already in
`benchmarks/LOCA-Bench/`. Follow [the installation commands](../benchmarks/LOCA-Bench/README.md)
for Python, Node/npm, and MCP executables. No dataset download is required:
the environment generates task workspaces from its included code and seed assets.

Set `loca_environment_length` to `8k`, `16k`, `32k`, `64k`, `96k`, `128k`, or
`256k`. Each length selects 75 cases: 15 task families with five seeds. To check
the included configuration files without starting tasks:

```bash
python scripts/prepare_benchmarks.py --benchmark loca
```

Tools execute code and change files. Use a disposable environment without
private files or production credentials.

## RCA100

```bash
pip install -r requirements/rca100.txt
python scripts/prepare_benchmarks.py --benchmark rca100
```

The command downloads the seven public observation files for each of 103 cases
into `benchmarks/RCA100/cases/`. The separately stored `answer_key/` is included
with confirmed redistribution authorization and is only used by the evaluator.
Original third-party notices remain in place.

To import an existing dataset directory containing `cases/`:

```bash
python scripts/prepare_benchmarks.py --benchmark rca100 --source /path/to/rca100
```

`manifest.txt` defines case order. It is needed by the adapter, not an extra
release-management mechanism. A different compatible authorized label directory
can be supplied with `--answer-key`.

## ClinDiag

Obtain the official archive from [ClinDiag](https://github.com/geteff1/ClinDiag)
under its upstream data terms, then run:

```bash
python scripts/prepare_benchmarks.py --benchmark clindiag --source /path/to/Clindiag_Benchmark.zip
```

Preparation retains 302 valid rare cases and samples 302 challenging cases
deterministically with seed 42. The resulting order must match the included
`manifest.txt`. Observations go into `cases/`, while final diagnoses go into
`answer_key/` and are never agent evidence. An already prepared directory can
also be supplied through `--source`.

For any benchmark, add `--verify-only` to check required files without downloads.
Data access rights and upstream licenses remain separate from the PoS MIT license.
