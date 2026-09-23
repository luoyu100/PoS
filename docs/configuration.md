# Configuration

All entry points are standalone YAML files. There is no configuration inheritance
or automatic provider substitution.

## Shared fields

| Field | Meaning |
| --- | --- |
| `benchmark` | `ALFWorld`, `LOCA-Bench`, `RCA100`, or `ClinDiag` |
| `harness` | `ReAct` |
| `method` | Output label; select behavior with `context.type` |
| `context.type` | `raw` or `belief` |
| `start_case`, `end_case` | Half-open range in the selected benchmark ordering |
| `output_run_name` | Explicit directory label; do not reuse for incompatible settings |
| `max_steps` | Maximum policy decision turns, not necessarily tool calls |
| `skip_existing` | Reuse complete case results, including unsuccessful outcomes |
| `run_until_complete` | Restart crashed workers until the selected range has results |
| `restart_delay_seconds` | Delay between worker restarts |
| `include_actions` | Include the adapter's action schemas in the policy prompt |
| `max_json_retries` | Policy JSON repair limit; PoS internal repair is capped separately |

All model blocks use `model`, `base_url`, `api_key_env`, `temperature`,
`max_tokens`, and optional `extra_body`. Keep provider-specific parameters only
when the selected endpoint supports them. A missing key must be supplied through
the environment; do not put credentials in YAML.

## PoS fields

| Field under `context` | Meaning |
| --- | --- |
| `task_mode` | `execution` uses semantic progress scoring; `diagnostic` uses confidence distance |
| `update_every_steps` | Update cadence; public templates preserve one update per action |
| `agent_view` | Public templates use the compact text plus structured `hybrid` view |
| `model` | Belief initialization, candidate updates, and Belief Text generation |
| `diagnostic_progress_threshold` | Strict threshold for the half-L1 inferred-confidence distance |
| `sentinel.enabled` | Master Sentinel switch; not the consistency-only ablation |
| `sentinel.consistency_enabled` | Controls candidate consistency auditing only |
| `sentinel.progress_enabled` | Controls progress scoring independently |
| `sentinel.local_audit_every_steps` | Local audit cadence in Belief updates |
| `sentinel.global_audit_every_steps` | Global audit cadence; replaces local auditing on that update |
| `sentinel.recent_transition_count` | Recent evidence window for local audits |
| `sentinel.consistency_model` | Consistency caller; falls back to `sentinel.model` |
| `sentinel.progress_model` | Execution progress caller; falls back to the consistency model |
| `trapping.enabled` | Compute trapping diagnoses from the verified progress window |
| `trapping.window_size` | Number of verified transitions in the window |
| `trapping.recurrence_epsilon` | Maximum distance for a recurrence match |
| `trapping.max_cycle_length` | Maximum recurrence lag |
| `trapping.health_threshold` | Trapping when health is at or below this value; default 0.25 |
| `trapping.cycle_threshold` | Minimum recurrence for Cycle classification; default 0.75 |
| `recovery.enabled` | Generate constraints after a trapping diagnosis |
| `recovery.recent_low_progress_count` | Number of low-progress transitions included in recovery context |

Use a window of at least two transitions. Do not disable the Sentinel master
switch to approximate the consistency ablation: doing so also removes progress
needed for health estimation. The two public ablations are checked against full
PoS and differ only in the specified switches and output name.

## Benchmark parameters

- **ALFWorld:** `split: eval_out_of_distribution` uses `valid_unseen`; `task_types`
  selects the six official task types. The prepared manifest covers this split.
- **LOCA-Bench:** `loca_environment_length` chooses a task-config file;
  `loca_task_names: all` and the five configured seeds select 75 cases.
  `loca_max_tool_uses` is independent of `max_steps`;
  Raw and PoS use the same role-grouped tool catalog, preserving complete
  descriptions and parameter schemas without compression.
  `loca_tool_timeout_seconds` bounds a tool call. `loca_npm_cache` overrides the
  default `$XDG_CACHE_HOME/pos/npm` (or `~/.cache/pos/npm`). The adapter explicitly
  passes that path to MCP subprocesses.
- **RCA100:** case order follows `benchmarks/RCA100/manifest.txt`. Evaluator labels
  remain under `answer_key`, outside the tool-accessible observations.
- **ClinDiag:** `clindiag_subsets: all` uses rare then challenging cases;
  `clindiag_provider_model` serves environment information and
  `clindiag_judge_model` scores diagnoses. These roles are not PoS Sentinel roles.

| LOCA length | Default task-config path below `benchmarks/LOCA-Bench` |
| --- | --- |
| 8k | `task-configs/final_8k_set_config.json` |
| 16k | `task-configs/final_16k_set_config.json` |
| 32k | `task-configs/final_32k_set_config.json` |
| 64k | `task-configs/final_64k_set_config.json` |
| 96k | `task-configs/final_96k_set_config.json` |
| 128k | `task-configs/final_128k_set_config.json` |
| 256k | `task-configs/final_256k_set_config.json` |

Leave `loca_task_config` unset to use this mapping. If explicitly provided, it
overrides the inferred file and must be changed together with the length.

## Sharding

Copy a configuration to your own local file and change only the range when
sharding a run. For 103 RCA100 cases, `[0,25)`, `[25,50)`, `[50,75)`, and
`[75,103)` cover the complete range without overlap. Keep the same output name
and all other method settings. Result aggregation includes completed cases
already in that run; do not mix incompatible configurations in one directory.

The legacy zero-valued `compressor` token bucket remains in result summaries for
schema compatibility. No compression baseline is implemented in this release.
