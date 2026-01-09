\# Review Priority Score (RPS)



RPS is a deterministic triage score (0–100) that prioritizes engineering review of OT telecom export data.



\## Design goals

\- Deterministic: same input to same output

\- Explainable: no ML required

\- Stable: large datasets don’t cause runaway scores

\- Auditable: score includes driver breakdown



\## Inputs

\- `errors`: count of ERROR issues

\- `warnings`: count of WARNING issues

\- `orphaned\_device`: count of orphaned devices

\- `single\_point\_of\_failure`: 1 if any SPOF exists else 0

\- `doc\_confidence`: documentation confidence in \[0,1]

\- `doc\_staleness = 1 - doc\_confidence`



\## Normalization (log scaling)

Counts are mapped to \[0,1] using a saturating log transform:



lognorm(x; k) = log(1 + x) / log(1 + k), clamped to \[0,1]



This prevents one metric from dominating as counts grow.



\## Components

\- E = lognorm(errors, k=5)

\- W = lognorm(warnings, k=10)

\- O = lognorm(orphaned\_device, k=3)

\- P = 1 if SPOF present else 0

\- S = doc\_staleness



\## Weighted sum

Weights sum to 1 for interpretability (convex combination):



r = 0.40\*E + 0.20\*O + 0.15\*P + 0.15\*S + 0.10\*W  

RPS = 100 \* r



\## Levels

\- HIGH: RPS >= 70

\- MEDIUM: 40 <= RPS < 70

\- LOW: RPS < 40



\## Output

The JSON report exports:

\- `review\_priority\_score`

\- `review\_priority`

\- `review\_priority\_drivers` (raw counts, normalized components, and weights)



