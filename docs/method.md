# Method implementation

## Runtime loop

PoS maintains the structured belief \(B_t=(W_t,G,\Delta_t^E,\Delta_t^A)\).
The world uses Entities, States, and Relations. The Active Gap identifies the
current focus, while health and recovery are runtime bookkeeping.

1. Initialize the belief and Active Gap from the task and initial observation.
2. The policy reads compact Belief Text, the relevant structured subgraph, the
   latest observation, and any recovery constraint.
3. Execute an action and construct a candidate belief update from its observation.
4. The synchronous Sentinel validates internal and evidence consistency. A detected
   issue permits one repair and re-audit. A rejected candidate does not replace
   the committed belief.
5. Score progress against the finalized belief transition. A scoring failure stays
   unverified and does not become a zero-progress label.
6. If the environment is terminal, do not generate further recovery constraints.
7. Keep the Active Gap unless resolved, estimate health, and compose or release
   recovery constraints for the next action.

RCA permits several independent actions in one decision. They execute in order,
with a belief update for each action. Policy turns and executed actions are
separately budgeted; logged transition steps are zero-based action indices.

After authoritative environment termination, context-update or context-export
errors are recorded without discarding the environment's result. This applies to
both successful and unsuccessful terminal results in the shared Raw/PoS loop;
nonterminal errors still propagate. No additional environment action is taken.

## Progress

For diagnostic tasks, Python aligns inferred States and Relations by type and
stable record ID. Confidence values are independent and are not normalized.
Missing-side confidence is zero.

\[
d_t^{diag}=\tfrac12\sum_x|c_t(x)-c_{t-1}(x)|,
\qquad u_t=\mathbb I[d_t^{diag}>\epsilon].
\]

The sum can exceed one. The task-conditioned world supplies the relevant records;
there is no additional learned relevance model. For execution tasks, Sentinel
labels progress when evidence reduces the Active Gap, provides needed information,
or advances a plausible path to the goal. Authoritative task success is progress.

## Health estimation

Use the latest \(K\) transitions with verified progress and their committed
snapshots. Unverified transitions are excluded; frontier changes do not reset the
window. Stable gap matching uses normalized targets and conservative token overlap.
Persistence measures the fraction of the window's initial gaps that persist:

\[
P_X=\frac{|\bigcap_{i=1}^{K}\Delta_i^X|}{|\Delta_1^X|},
\qquad S=1-\frac1K\sum_{i=1}^{K}u_i.
\]

An empty initial gap set gives zero persistence. All historical worlds are
projected onto the same current Active Gap using text matching and existing
relations. Canonical record signatures include confidence and stable identity;
field order does not change the signature. Text matching is an implementation
approximation, not a learned semantic equivalence guarantee.

\[
d_W(W_i^\Delta,W_j^\Delta)=\tfrac13\sum_{Y\in\{E,S,R\}}
d_J(Y_i^\Delta,Y_j^\Delta),
\]
\[
R=\max_{1\leq\ell\leq L}\frac1{K-\ell}
\sum_{i=\ell+1}^{K}\mathbb I[d_W(W_i^\Delta,W_{i-\ell}^\Delta)\leq\epsilon_W],
\qquad 1\leq L<K,
\]
\[
H=1-\max(P_E,P_A)\max(S,R).
\]

Trapping requires \(H\leq\theta_H\), with default \(\theta_H=0.25\).
On recurrence ties, the smaller positive lag is selected.

## Diagnosis and recovery

Evaluate these conditions in order:

1. Static: the last two complete worlds have distance at most \(\epsilon_W\).
2. Cycle: \(R\geq\theta_R\) and the dominant lag exceeds one; \(\theta_R=0.75\).
3. Drift: the last two Active Gap projections have distance at most \(\epsilon_W\).
4. Otherwise, retain an unclassified pattern and provide generic recovery.

Select the higher-persistence type among nonempty current gap sets. Epistemic
has priority on a tie. Compose a pattern constraint and a gap-type constraint:

\[
C=C_{pattern}\cup C_{gap}.
\]

Static suppresses the ineffective state-action transition; Cycle supplies matched
state-action transitions to help the policy break recurrence; Drift re-anchors to
the Active Gap. Epistemic constraints request discriminative evidence; Achievement
constraints request a relevant world change. Recovery never replaces the Active
Gap or executes an action directly. After each scored nonterminal transition,
release constraints if health exceeds the threshold; otherwise refresh them from
the current diagnosis. A resolved Active Gap advances through the normal selector.

The recovery target is an unresolved gap of the diagnosed blocked type, preferring
gaps persistent throughout the health window. Within these candidates the Active
Gap is retained when applicable; otherwise the first candidate in belief order is
used. The recovery target is stored separately from the unchanged Active Gap.

## Implementation and observability

`trappingDetection.py` computes health and classification, `progressScoring.py`
computes diagnostic progress, `beliefSentinel.py` performs audits and execution
scoring, and `manager.py` commits beliefs and maintains the recovery lifecycle.
`recoveryPlanning.py` composes instructions without another model call.

Events and results retain raw observations, candidate and committed beliefs,
progress, health components, the Active Gap, and recovery constraints. Markdown
renders the same health score and threshold. The public LOCA configurations use
100 policy turns and 100 tool calls. Model validation tests do not substitute for
full benchmark execution.
