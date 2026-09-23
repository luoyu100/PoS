"""Info utilities."""

from __future__ import annotations


FAULT_TYPE_IDS: list[str] = [
    "nodeDown",
    "threadExhaustion",
    "trafficHotspot",
    "messageQueueBacklog",
    "trafficSurge",
    "memoryPressure",
    "cacheBreakdown",
    "slowSQL",
    "codeDefect",
    "cpuDeadLoop",
    "httpError5xx",
    "rateLimiting",
    "dbNetworkLatency",
    "loadBalancerFailure",
    "fullGC",
    "nullPointerException",
    "diskIOHigh",
    "nodeCpuHigh",
    "redisUnavailable",
    "nodeMemoryOOM",
    "slowResource",
    "cpuFullLoad",
    "replicaScaleDown",
    "resourceLimitMisconfig",
    "podCrashLoop",
    "podPendingUnschedulable",
    "podRestartFlapping",
    "networkPolicyIsolation",
    "dnsResolutionFailure",
]


FAULT_TYPES_INFO: str = """Fault type catalog (submit exactly one fault_type name from this closed set):
- nodeDown: A physical or virtual machine goes down or becomes unreachable, taking down every pod it hosts.
- threadExhaustion: All threads in the application thread pool are occupied, so new requests queue up or time out.
- trafficHotspot: Requests concentrate on one key, product, or endpoint, overloading a local partition or pod.
- messageQueueBacklog: Queue consumption falls behind production, so messages pile up and delivery latency grows.
- trafficSurge: External traffic rises sharply in a short time, exceeding the system's capacity.
- memoryPressure: Memory usage approaches the limit and triggers frequent GC; the application slows down without OOM. Bursty, occasional GC spikes, unlike the sustained high-frequency GC of fullGC.
- cacheBreakdown: Cache invalidation or misses let a flood of requests penetrate to the backend database or downstream services.
- slowSQL: Database queries take too long, causing upstream requests to pile up and time out.
- codeDefect: Other code bugs, such as missing parameter validation, logic errors, or configuration errors.
- cpuDeadLoop: Code falls into an infinite loop, pinning a single CPU core at 100% with one hot method, unlike the even multi-core load of cpuFullLoad.
- httpError5xx: A downstream service returns HTTP 5xx and the evidence is not yet enough to converge on a more specific root cause; a fallback type.
- rateLimiting: A downstream service triggers rate limiting, throttling, quota protection, or protective rejection; typically HTTP 429, sometimes a 403 with throttling semantics.
- dbNetworkLatency: Network latency, jitter, or packet loss on the path to a critical dependency (including the database) makes remote calls much slower.
- loadBalancerFailure: Kubernetes Service endpoints are broken or the load balancer is misconfigured, so traffic cannot be forwarded correctly.
- fullGC: JVM full GC fires continuously at high frequency (not occasional spikes); stop-the-world pauses make the application unavailable for long periods, unlike the bursty GC pressure of memoryPressure.
- nullPointerException: A specific code defect where a null reference is dereferenced (e.g. Java NPE), causing endpoint exceptions or 500s.
- diskIOHigh: Abnormal disk IOPS or latency slows down read and write operations.
- nodeCpuHigh: Host node CPU usage is too high, so multiple pods on that node are throttled together; a node-level fault rather than a single-pod issue.
- redisUnavailable: Redis or a cache instance is unavailable, unreachable, or keeps refusing connections, so dependent cache calls fail as a whole.
- nodeMemoryOOM: Host physical memory is exhausted and the kernel OOM killer starts killing processes.
- slowResource: Frontend static resources load slowly or service response time rises significantly; often appears as a propagation or impact node of an upstream bottleneck.
- cpuFullLoad: Application CPU runs high evenly across multiple cores (not a single-core dead loop), caused by heavy computation, external pressure, or concurrency.
- replicaScaleDown: Deployment replicas are scaled down to 0, all pods terminate, endpoints become empty, and traffic stops completely.
- resourceLimitMisconfig: Container memory/CPU limits are configured too low, so pods hit OOMKilled or CPU throttling under normal load; they fail again shortly after restart and the limit values are clearly low.
- podCrashLoop: Containers exit abnormally right after starting; the pod stays in CrashLoopBackOff and its restart count grows fast.
- podPendingUnschedulable: Pods stay Pending for a long time because nodeSelector, taints, resources, or volume conditions cannot be satisfied.
- podRestartFlapping: Pods start briefly but exit and restart periodically; the restart count keeps climbing and the service is intermittently available.
- networkPolicyIsolation: A NetworkPolicy or equivalent traffic policy blocks specific service-to-service communication; the services themselves run, but cross-service calls are rejected or time out.
- dnsResolutionFailure: DNS resolution of a Service name fails inside pods or resolves to a wrong address, breaking cross-service calls."""


DIAGNOSTIC_METHOD_INFO: str = """Diagnostic methodology (a general procedure, not an answer):
1. Scope the incident. Read the alert window and the alert entity from the task, then call get_data_overview once to learn which modalities exist and what time range they cover. get_alerts shows the alert lifecycle and the triggering values.
2. Build the dependency picture. Use get_topology on the alert entity to find its callers, callees, and the pod/node that hosts it. list_entities gives the verbatim names of services and nodes. The alert entity is where the symptom surfaced, so treat every upstream dependency as a root-cause candidate until the evidence rules it out.
3. Discover metric names before querying them. Metric names in this dataset are fixed strings that cannot be guessed from convention: call list_metric_names (optionally with a keyword such as gc, thread, cpu, mem, node, error, replicas) and copy a name verbatim into query_metrics. A guessed name wastes a tool call and returns nothing.
4. Quantify each candidate against its own baseline. query_metrics returns a per-minute series plus overall min/mean/max, so compare the value inside the alert window with the quiet minutes before it. A conclusion needs a concrete change (a multiple, a percentage, or a level crossing), not merely a metric that looks high.
5. Cross-check with a second modality. search_logs (pod_name prefix plus keyword) shows application errors, query_traces with error_only or min_duration_ms shows failing or slow spans and their status messages, and get_k8s_events shows pod and node lifecycle reasons such as OOMKilled, Killing, Unhealthy, FailedScheduling, or scaling. One modality alone is rarely enough to separate two similar fault types.
6. Separate the origin from the propagation. A latency or error increase that appears in several services along one call path usually has a single origin: the service whose own resource, dependency, or code signal degrades first and whose degradation is not explained by any of its own dependencies. Do not stop at the first service that merely reports errors received from downstream.
7. Always test the infrastructure layer before concluding a service-level fault. Check the pods and the hosting node of the suspect service (node_cpu_usage_rate, node_memory_usage_rate, node_disk_usage_rate, node_ready_status, node_pod_running_count) together with k8s events. When several unrelated pods on the same node degrade at once, the root cause is the node, not any single service. When only one workload degrades, stay at the service level.
8. Discriminate inside the fault catalog before submitting. Several types share a symptom and differ only in a specific, checkable signal. Take the shortlist of types consistent with the symptom and verify the discriminating signal for each one, for example:
   - application CPU high: cpuDeadLoop concentrates on one core or one hot method, cpuFullLoad rises evenly across cores, nodeCpuHigh is a node-wide rise affecting co-located pods, and trafficSurge or trafficHotspot shows a matching rise in request_count or workload.
   - JVM memory symptoms: read the GC metrics (arms_jvm_gc_delta, arms_jvm_gc_seconds_delta, arms_jvm_mem_used_bytes vs arms_jvm_mem_max_bytes). fullGC means GC fires continuously at high frequency over the whole window with long stop-the-world pauses; memoryPressure means bursty or occasional GC spikes while the application keeps serving and is never OOM-killed; nodeMemoryOOM means the kernel OOM killer acted on the host; resourceLimitMisconfig means the container limit itself is clearly too low and the pod is OOMKilled again soon after restart.
   - errors returned by a service: rateLimiting shows throttling or quota semantics, typically HTTP 429 or a 403 with throttling wording; nullPointerException shows an explicit null-dereference stack; httpError5xx is the correct answer when the service clearly returns 5xx and the evidence does not converge on one of the more specific mechanisms above.
   - slow dependency calls: slowSQL shows long database query spans, dbNetworkLatency shows added latency or jitter on the path to the dependency without the query itself being slow, redisUnavailable shows cache calls failing outright, and cacheBreakdown shows cache misses pushing load onto the backend.
   - pods not serving: replicaScaleDown shows desired replicas at 0 and empty endpoints, podCrashLoop shows containers exiting right after start with CrashLoopBackOff, podRestartFlapping shows periodic restarts with intermittent availability, podPendingUnschedulable shows pods stuck Pending with a scheduling reason, and nodeDown shows the whole node unreachable.
   - calls failing while both services are healthy: networkPolicyIsolation shows connections rejected or timed out for specific service pairs, dnsResolutionFailure shows name resolution errors, and loadBalancerFailure shows broken or misconfigured Service endpoints.
9. Use the turn budget. Reaching a plausible answer early is not a reason to submit: spend the remaining turns confirming the discriminating signal of your chosen type and ruling out its nearest alternative, because a wrong fault type scores the same as a wrong entity. Submit once the discriminating evidence is in hand."""


ENTITY_STANDARD_INFO: str = """Root-cause entity standard:
- Submit exactly ONE root-cause entity.
- The entity must be at one of two levels of the microservice system:
  1. service level: the microservice name (entity type apm.service), e.g. "inventory", "cart", "frontend".
  2. node level: the Kubernetes node name (entity type k8s.node), e.g. "cn-hongkong.10.0.1.69".
- Use the entity name verbatim as it appears in the observability data returned by the tools (topology, metrics, logs, traces, events). Do not invent, translate, or abbreviate names.
- Do not submit pod names, instance names, operation names, or any other entity level; they are treated as a localization error.
- The alert entity in the task is only the symptom entry point; it is frequently NOT the root cause. Investigate before submitting."""


SUBMIT_CONTRACT_INFO: str = """Submission contract:
- Investigate with the available tools first; the topology and observability data are NOT included in the task input and must be read through tools.
- When the evidence is sufficient, call the tool "submit_root_cause" exactly once with arguments:
  {"root_cause_entity": "<one entity name>", "fault_type": "<one fault type name from the catalog>", "analysis": "<brief evidence-backed reasoning>"}
- fault_type must be exactly one name from the fault type catalog, e.g. "nodeMemoryOOM"; any other value is rejected.
- A valid submission ends the case immediately; there is no second attempt after a valid submission.
- An invalid submission (unknown fault_type, missing or empty root_cause_entity) is rejected with an error observation and consumes one step; fix it and submit again."""
