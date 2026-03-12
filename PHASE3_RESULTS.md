# TCA Phase 3 Results: Full KD Data at Scale

## Data

| Metric | Value |
|--------|-------|
| Total KD nodes loaded | 1,430 (930 facts + 500 beliefs) |
| Unique domains | 99 |
| Unique tags | 479 |
| Top domains | cognitive-architecture (147), infrastructure (143), saturn (101) |

## Graph

| Metric | Value |
|--------|-------|
| Nodes | 1,430 |
| Edges (stored) | 71,102 |
| Edges (bidirectional) | 142,204 |
| Average edges per node | 98.4 |
| INHERITS edges | 32,091 (shared domain) |
| MIRRORS edges | 37,487 (shared tags) |
| EXPRESSES edges | 1,524 (content cross-references) |

## Grounding Results

| Metric | Value |
|--------|-------|
| Nodes grounded | 50 (from 3 domains) |
| Domains | cognitive-architecture, infrastructure, saturn |
| Edges checked | 2,892 |
| Edges validated | 2,892 (100%) |
| Edges weakened | 0 |
| Grounded edges before | 0 |
| Grounded edges after | 4,820 |
| Grounding ratio | 3.39% |
| Avg source confidence | 1.000 |

## Full-Scale Benchmark: Chestohedron vs NULL

5 queries, each run with chestohedron router and NULL router (all gates = 1/7).

| Query | Router | Ticks | Nodes | Domains | Bridges | Cycles | Conf | Ground |
|-------|--------|-------|-------|---------|---------|--------|------|--------|
| Q1: What is KD? | CHESTO | 2 | 345 | 35 | 0 | 4,251 | 0.192 | 0.000 |
| | NULL | 2 | 345 | 35 | 0 | 7,820 | 0.192 | 0.000 |
| Q2: Reservoir computing | CHESTO | 2 | 371 | 38 | 0 | 3,531 | 0.196 | 0.000 |
| | NULL | 2 | 371 | 38 | 0 | 3,531 | 0.196 | 0.000 |
| Q3: Topology + consciousness | CHESTO | 2 | 366 | 42 | 0 | 2,476 | 0.192 | 0.000 |
| | NULL | 2 | 366 | 42 | 0 | 2,476 | 0.192 | 0.000 |
| Q4: Chestohedron + economics | CHESTO | 2 | 293 | 32 | 0 | 1,814 | 0.192 | 0.000 |
| | NULL | 2 | 293 | 32 | 0 | 2,558 | 0.192 | 0.000 |
| Q5: TRINITY + CCN | CHESTO | 2 | 271 | 32 | 0 | 17,296 | 0.204 | 0.000 |
| | NULL | 2 | 271 | 32 | 0 | 17,296 | 0.204 | 0.000 |

**Averages:**

| Router | Ticks | Nodes | Domains | Bridges | Cycles | Conf | Ground |
|--------|-------|-------|---------|---------|--------|------|--------|
| CHESTOHEDRON | 2.0 | 329.2 | 35.8 | 0.0 | 5,873.6 | 0.195 | 0.000 |
| NULL | 2.0 | 329.2 | 35.8 | 0.0 | 6,736.2 | 0.195 | 0.000 |

**Prediction: Chestohedron outperforms NULL on Q3-Q5. Result: WRONG.**

All 5 queries tied on confidence. Chestohedron showed slightly fewer cycles on Q1 and Q4, but no measurable advantage on confidence, node activation, domain coverage, or bridges.

## Parameter Count

| Layer | Parameters | Description |
|-------|-----------|-------------|
| L1 Router | 70 | 69 keyword patterns + 1 baseline |
| L3 Grounding | 4 | learning rate, max/min weight, delta threshold |
| L4 Temporal | 3 | max_ticks, convergence_threshold, decay_factor |
| L5 Confidence | 4 | confidence weights |
| **Total** | **81** | **Invariant to graph size** |

81 parameters for 1,430 nodes — identical to Phase 2's 81 parameters for 30 nodes.

## Did the gap open up?

No. The chestohedron router showed zero advantage over NULL routing at full scale. This is the opposite of the Phase 2 prediction that "router differentiation emerges at 1,000+ nodes with longer activation paths." The graph is too densely connected: 98.4 average edges per node means activation floods the entire graph in 2 ticks regardless of which gates are weighted. Both routers activate the same ~330 nodes, touch the same ~35 domains, and reach identical confidence scores. The routing signal — which gates to emphasize — is drowned out by the sheer density of connections. The chestohedron's selective gate weighting only matters when the graph is sparse enough that different edge types lead to genuinely different subgraphs. At 142K edges connecting 1,430 nodes, every path leads everywhere.

## Did pure topology hold at scale?

The architecture held — no crashes, no embeddings, all 1,430 nodes loaded and reasoned over in under 6 seconds, and the 81-parameter invariance is confirmed. What didn't hold is the claim that topology-selective routing produces better results than uniform routing. The system works, but the router doesn't matter at this density. The honest next step is either (a) sparser edge construction with stricter thresholds, so the topology actually constrains activation flow, or (b) a fundamentally different activation strategy that doesn't flood a dense graph in 2 ticks. The 81-parameter count is real. The topology-over-substance thesis needs a graph where topology actually varies by path.
