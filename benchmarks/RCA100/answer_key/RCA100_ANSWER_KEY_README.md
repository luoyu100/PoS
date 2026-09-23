# RCA-100 v1.1 — Answer Key Package

Updated for RCA-100 v1.1 by **Alibaba Cloud AIOps Team**. Contains ground-truth annotations for all 103 cases.

## Contents (106 files, including this README)

- `mapping.json` — task_id (t001..t103) → case_id mapping
- `taxonomy.json` — fault hierarchy and partial-credit scoring rubric
- `t001.gt.json` ... `t103.gt.json` — per-case ground truth (each ~5–9 KB)

## Per-case GT schema (top-level fields)

- `incident_id` — canonical brise incident ID
- `case_id` — same as incident_id (legacy alias)
- `workspace`, `region_id` — provenance
- `time_range_start`, `time_range_end` — incident window (UTC)
- `root_cause_entities[]` — list of GT root-cause entity names
- `alert_event_id`, `alert_title`, `alert_trigger_time` — triggering alert
- `alert_entity_id`, `alert_entity_type`, `alert_entity_domain` — normalized alert entity fields where available
- `prompt_text` — task prompt (handed to agent at evaluation)
- `raw_ground_truth` — structured fault chain + canonical RCA narrative

## Use

- For agent scoring: do NOT include this in the prompt context. Score against `root_cause_entities` and `raw_ground_truth.fault_type` only.
- For training: cite RCA-100 v1.1 and link to the public dataset URL.
- For redistribution: please contact the Alibaba Cloud AIOps Team before sharing publicly.

## Public dataset (no answer key)

- Cases: `oss://aiops-benchmark/rca/rca100/v1.1/cases/` (publicly readable)
- Root docs: `README.md`, `manifest.txt`, `summary.json` at `/rca/rca100/v1.1/` (publicly readable)
- This answer_key/ folder is **no longer publicly readable** as of 2026-05-26; distribute through controlled channels only.
