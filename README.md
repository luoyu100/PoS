# PoS

> Anonymous code release for review.

## 📖 Overview

PoS maintains a structured Belief State to guide long-horizon agent decisions.
It represents the current world through entities, states, and relations, tracks
task-relevant gaps, and uses consistency validation and trapping diagnosis to
support reliable reasoning and recovery.

The code supports **ALFWorld**, **LOCA-Bench**, **RCA100**, and **ClinDiag**, with
Raw/ReAct as the baseline and two PoS ablations.

## 🌟 Key Features

- **Belief construction:** maintain a task-conditioned world representation from observations.
- **Active Gap:** focus the next decision on an unresolved task-relevant gap.
- **Consistency validation:** check candidate updates against the available evidence.
- **Trapping diagnosis and recovery:** detect stalled or recurrent reasoning and provide recovery constraints.
- **Unified evaluation:** switch benchmarks and model roles through YAML configurations.

## 🛠️ Installation

1. Download and extract this repository, then enter its root directory.

2. Create the environment. Linux with Python 3.11 is the target platform.

   ```bash
   conda create -n PoS python=3.11
   conda activate PoS
   pip install -r requirements.txt
   ```

3. Install the dependencies for the benchmarks you need.

   ```bash
   pip install -r requirements/alfworld.txt
   pip install -r requirements/loca.txt
   pip install -r requirements/rca100.txt
   ```

   ClinDiag uses the core dependencies. LOCA also requires Node.js/npm and MCP
   tools; follow [its setup instructions](benchmarks/LOCA-Bench/README.md).

4. Configure your API key.

   ```bash
   export DASHSCOPE_API_KEY="YOUR_API_KEY"
   ```

   The default backbone is Qwen3.7-Plus. Edit `model`, `base_url`, and
   `api_key_env` in the YAML to use another provider. The policy model and the
   Belief/Sentinel models are configured separately.

## 🚀 Quick Start

### Prepare benchmark data

Runtime integration is included under `benchmarks/`. Large datasets are not
bundled; download or import them with the commands below.

```bash
python scripts/prepare_benchmarks.py --benchmark alfworld
python scripts/prepare_benchmarks.py --benchmark rca100
python scripts/prepare_benchmarks.py --benchmark clindiag --source /path/to/Clindiag_Benchmark.zip
```

LOCA's runtime and task configurations are already included. It generates task
workspaces when an experiment starts; no separate dataset download is needed.
RCA100 evaluator labels are included and kept separate from agent-visible data.
See [data setup](docs/data.md) for local imports and prerequisites.

### Run PoS

```bash
python main.py --config configs/alfworld/pos.yaml
python main.py --config configs/loca/pos.yaml
python main.py --config configs/rca100/pos.yaml
python main.py --config configs/clindiag/pos.yaml
```

| Benchmark | Default evaluation range |
| --- | --- |
| ALFWorld | 134 valid-unseen tasks |
| LOCA-Bench | 75 tasks at 8k |
| RCA100 | 103 cases |
| ClinDiag | 604 cases |

For LOCA, set `loca_environment_length` to `8k`, `16k`, `32k`, `64k`, `96k`,
`128k`, or `256k`. For a first trial, set `start_case: 0`, `end_case: 1`, and
`run_until_complete: false` in a separate configuration.

### Baseline and ablations

Each benchmark has the same four configuration files:

| Configuration | Method |
| --- | --- |
| `pos.yaml` | Full PoS |
| `raw.yaml` | Raw trajectory / ReAct |
| `pos_no_consistency.yaml` | PoS without consistency validation |
| `pos_no_trapping.yaml` | PoS without trapping diagnosis and recovery |

```bash
python main.py --config configs/rca100/raw.yaml
python main.py --config configs/rca100/pos_no_consistency.yaml
python main.py --config configs/rca100/pos_no_trapping.yaml
```

Case ranges are `[start_case, end_case)`. `skip_existing: true` reuses completed
results, including failed cases; `run_until_complete: true` restarts crashed
workers. Model calls incur provider costs.

`max_steps` counts policy turns, which may contain multiple tool actions. LOCA
also has `loca_max_tool_uses`: all four LOCA templates use 100 tool calls and
100 policy turns. Keep these limits equal when comparing methods.

## 📂 Project Structure

```text
PoS/
├── belief/                 # Belief Manager, Sentinel, trapping, and recovery
├── baselines/              # Shared ReAct agent loop
├── contexts/               # Raw trajectory and context interface
├── benchmarks/
│   ├── adapter.py          # Unified benchmark adapter
│   ├── ALFWorld/           # ALFWorld data setup
│   ├── LOCA-Bench/         # Included LOCA runtime and task configurations
│   ├── RCA100/             # RCA tools, task order, and evaluator labels
│   └── ClinDiag/           # Clinical task instructions and fixed case order
├── configs/                # Four configurations per benchmark
├── prompts/                # Policy, Belief, and Sentinel prompts
├── utils/                  # Logging and Markdown rendering
├── scripts/                # Data setup and validation utilities
├── docs/                   # Configuration and implementation details
├── main.py                 # Experiment entry point
└── requirements.txt        # Core dependencies
```

## 📝 Logs and Reproducibility

Experiments write `events.jsonl`, `events.md`, and `result.json` under `outputs/`.
These record model interactions, Belief updates, progress, trapping, recovery,
and token usage. No previous experiment outputs are included in this release.

See [configuration](docs/configuration.md), [method implementation](docs/method.md),
and [validation notes](docs/development.md). Offline checks are not a substitute
for live provider and complete benchmark testing; the ClinDiag PoS template has
not been tested in a real-model run.

## License

Original PoS code uses the [MIT License](LICENSE). Benchmark code and data retain
their own [third-party terms](THIRD_PARTY_NOTICES.md). Run LOCA tools in an isolated
environment without personal files or production credentials.
