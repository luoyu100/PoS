"""Bounded observability queries for RCA100.

Tools lazily read metrics, logs, traces, events, alerts, and topology.
Query results never expose the separate evaluator answer key."""

from __future__ import annotations

import json

import pandas as pd

TZ = "Asia/Shanghai"


def _dumps(payload: object) -> str:
    """Dumps."""

    return json.dumps(payload, ensure_ascii=False, default=str)


def _iso_us(value) -> str:
    """Iso us."""

    return (
        pd.to_datetime(int(value), unit="us", utc=True)
        .tz_convert(TZ)
        .strftime("%Y-%m-%dT%H:%M:%S%z")
    )


def _iso_ns(value) -> str:
    """Iso ns."""

    return (
        pd.to_datetime(int(value), unit="ns", utc=True)
        .tz_convert(TZ)
        .strftime("%Y-%m-%dT%H:%M:%S%z")
    )


def _same(series: pd.Series, value: str) -> pd.Series:
    """Same."""

    return series.astype(str).str.lower() == str(value).strip().lower()


def get_data_overview(case) -> str:
    """Get data overview."""

    metrics = case.metrics()
    logs = case.logs()
    traces = case.traces()
    events = case.events()
    alerts = case.alerts()
    topology = case.topology()
    return _dumps(
        {
            "timezone": "+08:00",
            "metrics": {
                "rows": len(metrics),
                "entities": int(metrics["entity_name"].nunique()),
                "distinct_metrics": int(metrics["metric"].nunique()),
                "time_range": [
                    _iso_us(metrics["time"].min()),
                    _iso_us(metrics["time"].max()),
                ],
            },
            "logs": {
                "rows": len(logs),
                "pods": int(logs["_pod_name_"].nunique()),
                "time_range": [
                    str(logs["_time_"].min()),
                    str(logs["_time_"].max()),
                ],
            },
            "traces": {
                "rows": len(traces),
                "services": int(traces["serviceName"].nunique()),
                "time_range": [
                    _iso_ns(pd.to_numeric(traces["startTime"]).min()),
                    _iso_ns(pd.to_numeric(traces["startTime"]).max()),
                ],
            },
            "k8s_events": {"rows": len(events)},
            "alerts": {"rows": len(alerts)},
            "topology": {
                "entities": len(topology["entities"]),
                "edges": len(topology["edges"]),
            },
        }
    )


def list_entities(case, entity_type, keyword=None, limit=50) -> str:
    """List entities."""

    limit = max(1, min(int(limit), 100))
    names = sorted(
        {
            entity["name"]
            for entity in case.topology()["entities"]
            if entity["type"] == entity_type
            and (keyword is None or str(keyword).lower() in entity["name"].lower())
        }
    )
    return _dumps(
        {
            "entity_type": entity_type,
            "matched": len(names),
            "returned": min(len(names), limit),
            "names": names[:limit],
        }
    )


def list_metric_names(case, entity_name=None, keyword=None, limit=80) -> str:
    """List metric names."""

    limit = max(1, min(int(limit), 120))
    frame = case.metrics()
    if entity_name is not None:
        frame = frame[_same(frame["entity_name"], entity_name)]
        if frame.empty:
            return _dumps(
                {
                    "error": f"no metrics for entity {entity_name!r}; "
                    "check the exact name via get_topology or list_entities"
                }
            )
    if keyword is not None:
        frame = frame[
            frame["metric"]
            .astype(str)
            .str.contains(str(keyword), case=False, regex=False)
        ]

    counts = frame.groupby("metric")["entity_name"].nunique().sort_index()
    names = [
        {"metric": str(metric), "entities": int(total)}
        for metric, total in counts.items()
    ]
    payload = {
        "matched": len(names),
        "returned": min(len(names), limit),
        "metrics": names[:limit],
        "hint": (
            "Copy a metric name verbatim into query_metrics(entity_name, "
            "metric) to get its per-minute time series."
        ),
    }
    if entity_name is not None:
        payload = {"entity_name": entity_name, **payload}
    return _dumps(payload)


def get_topology(case, entity_name=None, relation=None, limit=30) -> str:
    """Get topology."""

    limit = max(1, min(int(limit), 50))
    topology = case.topology()
    entities = topology["entities"]

    if entity_name is None:
        counts: dict[str, int] = {}
        for entity in entities:
            counts[entity["type"]] = counts.get(entity["type"], 0) + 1
        return _dumps(
            {
                "entity_counts_by_type": dict(sorted(counts.items())),
                "services": sorted(
                    entity["name"]
                    for entity in entities
                    if entity["type"] == "apm.service"
                ),
                "nodes": sorted(
                    entity["name"]
                    for entity in entities
                    if entity["type"] == "k8s.node"
                ),
                "hint": (
                    "Call get_topology with entity_name to inspect one "
                    "entity and its neighbors."
                ),
            }
        )

    wanted = str(entity_name).strip().lower()
    matched = [entity for entity in entities if entity["name"].lower() == wanted]
    if not matched:
        similar = sorted(
            {entity["name"] for entity in entities if wanted in entity["name"].lower()}
        )[:10]
        return _dumps(
            {
                "error": f"entity {entity_name!r} not found in topology",
                "similar_names": similar,
            }
        )

    names_by_id = {entity["id"]: entity["name"] for entity in entities}
    matched_ids = {entity["id"] for entity in matched}
    edges = [
        {
            "relation": edge["relation"],
            "src": names_by_id.get(edge["src"], edge["src"]),
            "src_type": edge["src_type"],
            "dst": names_by_id.get(edge["dst"], edge["dst"]),
            "dst_type": edge["dst_type"],
        }
        for edge in topology["edges"]
        if (edge["src"] in matched_ids or edge["dst"] in matched_ids)
        and (relation is None or edge["relation"] == relation)
    ]
    return _dumps(
        {
            "entities": [
                {
                    "name": entity["name"],
                    "type": entity["type"],
                    "props": {
                        key: entity.get("props", {}).get(key)
                        for key in ("namespace", "node_name", "status")
                        if entity.get("props", {}).get(key) is not None
                    },
                }
                for entity in matched
            ],
            "edges_matched": len(edges),
            "edges": edges[:limit],
        }
    )


def query_metrics(case, entity_name, metric=None, limit=40) -> str:
    """Query metrics."""

    limit = max(1, min(int(limit), 60))
    frame = case.metrics()
    frame = frame[_same(frame["entity_name"], entity_name)]
    if frame.empty:
        return _dumps(
            {
                "error": f"no metrics for entity {entity_name!r}; "
                "check the exact name via get_topology or list_entities"
            }
        )

    if metric is None:
        summary = (
            frame.groupby("metric")["value"]
            .agg(["count", "min", "mean", "max"])
            .round(6)
            .reset_index()
            .sort_values("metric")
        )
        records = summary.head(60).to_dict(orient="records")
        return _dumps(
            {
                "entity_name": entity_name,
                "distinct_metrics": len(summary),
                "metrics_summary": records,
                "hint": (
                    "Call query_metrics with metric to get its per-minute time series."
                ),
            }
        )

    frame = frame[_same(frame["metric"], metric)]
    if frame.empty:
        available = sorted(
            case.metrics()[_same(case.metrics()["entity_name"], entity_name)]["metric"]
            .astype(str)
            .unique()
        )
        return _dumps(
            {
                "error": f"metric {metric!r} not found for {entity_name!r}; "
                "use one of available_metrics verbatim, or call "
                "list_metric_names to browse them",
                "available_metrics": available[:80],
            }
        )
    frame = frame.assign(
        minute=pd.to_datetime(frame["time"], unit="us", utc=True)
        .dt.tz_convert(TZ)
        .dt.strftime("%Y-%m-%dT%H:%M%z")
    )
    series = frame.groupby("minute")["value"].mean().round(6).reset_index()
    points = series.to_dict(orient="records")
    truncated = len(points) > limit
    if truncated:
        points = points[-limit:]
    return _dumps(
        {
            "entity_name": entity_name,
            "metric": metric,
            "overall": {
                "min": round(float(frame["value"].min()), 6),
                "mean": round(float(frame["value"].mean()), 6),
                "max": round(float(frame["value"].max()), 6),
            },
            "total_minutes": len(series),
            "returned_minutes": len(points),
            "truncated_to_latest": truncated,
            "series": points,
        }
    )


def search_logs(case, pod_name=None, keyword=None, limit=20) -> str:
    """Search logs."""

    limit = max(1, min(int(limit), 50))
    frame = case.logs()
    if pod_name is not None:
        frame = frame[
            frame["_pod_name_"]
            .astype(str)
            .str.lower()
            .str.startswith(str(pod_name).strip().lower())
        ]
    if keyword is not None:
        frame = frame[
            frame["content"]
            .astype(str)
            .str.contains(str(keyword), case=False, regex=False)
        ]
    matched = len(frame)
    frame = frame.sort_values("_time_").tail(limit)
    lines = [
        {
            "time": str(row["_time_"]),
            "pod": str(row["_pod_name_"]),
            "content": str(row["content"])[:300],
        }
        for _, row in frame.iterrows()
    ]
    return _dumps(
        {
            "matched": matched,
            "returned": len(lines),
            "note": (
                "latest lines only; narrow with pod_name/keyword if matched is large"
            ),
            "lines": lines,
        }
    )


def query_traces(
    case,
    service_name=None,
    error_only=False,
    min_duration_ms=None,
    limit=10,
) -> str:
    """Query traces."""

    limit = max(1, min(int(limit), 30))
    frame = case.traces()
    if service_name is not None:
        frame = frame[_same(frame["serviceName"], service_name)]
    if error_only:
        frame = frame[frame["statusCode"].astype(str) == "2"]
    frame = frame.assign(
        duration_ms=pd.to_numeric(frame["duration"], errors="coerce") / 1e6
    )
    if min_duration_ms is not None:
        frame = frame[frame["duration_ms"] >= float(min_duration_ms)]
    matched = len(frame)
    if matched == 0:
        return _dumps({"matched": 0, "spans": []})

    top = frame.sort_values("duration_ms", ascending=False).head(limit)
    spans = [
        {
            "start": _iso_ns(pd.to_numeric(row["startTime"])),
            "service": str(row["serviceName"]),
            "span": str(row["spanName"])[:80],
            "duration_ms": round(float(row["duration_ms"]), 3),
            "status_code": str(row["statusCode"]),
            "status_message": str(row["statusMessage"] or "")[:80],
        }
        for _, row in top.iterrows()
    ]
    return _dumps(
        {
            "matched": matched,
            "duration_ms_stats": {
                "mean": round(float(frame["duration_ms"].mean()), 3),
                "max": round(float(frame["duration_ms"].max()), 3),
            },
            "returned_slowest": len(spans),
            "spans": spans,
        }
    )


def get_k8s_events(case, keyword=None, limit=20) -> str:
    """Get k8s events."""

    limit = max(1, min(int(limit), 50))
    records = []
    for _, row in case.events().iterrows():
        try:
            payload = json.loads(row["eventId"])
        except (TypeError, ValueError):
            continue
        involved = payload.get("involvedObject", {})
        record = {
            "last_time": payload.get("lastTimestamp") or payload.get("eventTime"),
            "type": payload.get("type"),
            "reason": payload.get("reason"),
            "object": f"{involved.get('kind')}/{involved.get('name')}",
            "count": payload.get("count"),
            "message": str(payload.get("message") or "")[:200],
        }
        if keyword is not None:
            haystack = _dumps(record).lower()
            if str(keyword).lower() not in haystack:
                continue
        records.append(record)
    records.sort(key=lambda item: str(item["last_time"]), reverse=True)
    return _dumps(
        {
            "matched": len(records),
            "returned": min(len(records), limit),
            "events": records[:limit],
        }
    )


def get_alerts(case, limit=12) -> str:
    """Get alerts."""

    limit = max(1, min(int(limit), 30))
    frame = case.alerts().sort_values("time")
    matched = len(frame)
    records = []
    for _, row in frame.head(limit).iterrows():
        data = {}
        try:
            data = json.loads(row["data"])
        except (TypeError, ValueError):
            pass
        records.append(
            {
                "time": str(row["time"]),
                "status": str(row["status"]),
                "severity": str(row["severity"]),
                "current_value": data.get("currentValue"),
            }
        )
    return _dumps({"matched": matched, "returned": len(records), "alerts": records})


TOOLS = [
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "get_data_overview",
                "description": (
                    "Get row counts, time ranges, and entity/metric/service "
                    "counts of every observability modality. Use it first "
                    "to plan queries. Returns one compact summary object."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        "function": get_data_overview,
    },
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "list_entities",
                "description": (
                    "List entity names of one topology type, e.g. "
                    "apm.service, k8s.node, k8s.pod, k8s.deployment. Use it "
                    "to verify the exact spelling of an entity name. "
                    "Returns at most 100 names (default 50)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_type": {
                            "type": "string",
                            "description": (
                                "Topology entity type such as apm.service, "
                                "k8s.node, k8s.pod, k8s.deployment."
                            ),
                        },
                        "keyword": {
                            "type": "string",
                            "description": "Optional substring filter.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max names to return, <= 100.",
                        },
                    },
                    "required": ["entity_type"],
                },
            },
        },
        "function": list_entities,
    },
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "list_metric_names",
                "description": (
                    "List the metric names that actually exist in this "
                    "incident's data. Metric names are dataset-specific and "
                    "cannot be guessed, so call this before query_metrics. "
                    "Without entity_name: every metric name plus how many "
                    "entities report it. With entity_name: only that "
                    "entity's metric names. Returns at most 120 names "
                    "(default 80); narrow with keyword, e.g. gc, thread, "
                    "node, error, replicas."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_name": {
                            "type": "string",
                            "description": (
                                "Optional exact entity name; restricts the "
                                "list to metrics reported by that entity."
                            ),
                        },
                        "keyword": {
                            "type": "string",
                            "description": (
                                "Optional substring filter on the metric "
                                "name, e.g. gc, cpu, mem, thread, replicas."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max metric names to return, <= 120.",
                        },
                    },
                    "required": [],
                },
            },
        },
        "function": list_metric_names,
    },
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "get_topology",
                "description": (
                    "Without entity_name: entity counts by type plus the "
                    "full service and node name lists. With entity_name: "
                    "that entity's details and its relation edges "
                    "(contains/hosts/calls/...). Returns at most 50 edges "
                    "(default 30); narrow with the relation argument."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_name": {
                            "type": "string",
                            "description": "Exact entity name to inspect.",
                        },
                        "relation": {
                            "type": "string",
                            "description": (
                                "Optional edge relation filter, e.g. calls, "
                                "contains, hosts."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max edges to return, <= 50.",
                        },
                    },
                    "required": [],
                },
            },
        },
        "function": get_topology,
    },
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "query_metrics",
                "description": (
                    "Without metric: per-metric count/min/mean/max summary "
                    "for one entity (at most 60 metrics). With metric: its "
                    "per-minute time series, at most 60 latest minutes "
                    "(default 40) plus overall min/mean/max. Query one "
                    "entity and one metric at a time. The metric argument "
                    "must be a name that exists in this dataset; get it "
                    "from list_metric_names instead of guessing a "
                    "conventional name."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_name": {
                            "type": "string",
                            "description": (
                                "Exact entity name as it appears in "
                                "topology, e.g. a service, node, or pod."
                            ),
                        },
                        "metric": {
                            "type": "string",
                            "description": (
                                "Optional metric name, copied verbatim from "
                                "list_metric_names, e.g. "
                                "node_memory_usage_rate."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                "Max time-series minutes to return, <= 60."
                            ),
                        },
                    },
                    "required": ["entity_name"],
                },
            },
        },
        "function": query_metrics,
    },
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "search_logs",
                "description": (
                    "Search application logs by pod name prefix and/or "
                    "content keyword. Returns at most 50 latest matched "
                    "lines (default 20), each truncated to 300 characters. "
                    "Always narrow with pod_name or keyword; the log table "
                    "has hundreds of thousands of rows."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pod_name": {
                            "type": "string",
                            "description": (
                                "Pod name prefix, e.g. a service name like "
                                "'inventory' matches its pods."
                            ),
                        },
                        "keyword": {
                            "type": "string",
                            "description": (
                                "Case-insensitive substring, e.g. error, "
                                "OOM, timeout, exception."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max lines to return, <= 50.",
                        },
                    },
                    "required": [],
                },
            },
        },
        "function": search_logs,
    },
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "query_traces",
                "description": (
                    "Filter spans by service, error status, and minimum "
                    "duration, then return the slowest ones. Returns at "
                    "most 30 spans (default 10) plus matched count and "
                    "duration stats. Use filters; the span table has "
                    "hundreds of thousands of rows."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service_name": {
                            "type": "string",
                            "description": "Exact serviceName filter.",
                        },
                        "error_only": {
                            "type": "boolean",
                            "description": ("true keeps only spans with ERROR status."),
                        },
                        "min_duration_ms": {
                            "type": "number",
                            "description": (
                                "Keep spans at least this slow, in milliseconds."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max spans to return, <= 30.",
                        },
                    },
                    "required": [],
                },
            },
        },
        "function": query_traces,
    },
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "get_k8s_events",
                "description": (
                    "List Kubernetes events (reason, involved object, "
                    "message), newest first, with optional keyword filter. "
                    "Returns at most 50 events (default 20), messages "
                    "truncated to 200 characters."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": (
                                "Case-insensitive filter over reason, "
                                "object, and message, e.g. OOM, Failed, "
                                "a pod name."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max events to return, <= 50.",
                        },
                    },
                    "required": [],
                },
            },
        },
        "function": get_k8s_events,
    },
    {
        "schema": {
            "type": "function",
            "function": {
                "name": "get_alerts",
                "description": (
                    "Show the entry alert's lifecycle records (time, "
                    "status, severity, current value). Returns at most 30 "
                    "records (default 12)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Max records to return, <= 30.",
                        },
                    },
                    "required": [],
                },
            },
        },
        "function": get_alerts,
    },
]
