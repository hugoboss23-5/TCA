# TCA — Topological Cognitive Architecture
## Instruction Manual for AI Agents

You have access to TCA, a **deterministic structural analysis engine**. It contains ZERO AI, ZERO learned parameters. You (the AI) build a graph. TCA finds what's broken using pure topology.

---

## WHAT TCA DOES

TCA takes a directed graph of any system (company, country, economy, relationship, plan, religion, technology stack) and finds **structural problems** invisible to narrative reasoning:

- **Dead ends**: Nodes that absorb energy but produce nothing (e.g., "Human Wellbeing" in most economic models)
- **Feedback traps**: Circular loops with no exit (e.g., Debt → Interest → More Debt)
- **Star topologies**: Single node everything depends on — fragile (e.g., CEO, Fed, API provider)
- **Contradictions**: REMOVES edges — structural lies the system contains (e.g., "Revenue REMOVES Open Source")
- **Ungrounded claims**: SEEKS edges — things the system wants but hasn't proven

Then TCA **proposes solutions** by inverting each problem topologically.

---

## WHEN TO USE TCA

Call TCA when the user wants to understand the **structure** of something:
- "Analyze the structure of..."
- "What's broken about..."
- "Run TCA on..."
- "What are the contradictions in..."
- "Why does X keep failing?"
- Any system, organization, plan, ideology, business model, relationship dynamic
- Comparing two systems structurally
- Checking if a plan has hidden dependencies or single points of failure

## WHEN NOT TO USE TCA

Do NOT call TCA for:
- Simple factual questions ("What year was X founded?")
- Coding help or debugging
- Math calculations
- Creative writing
- Questions with a single correct answer
- Anything that doesn't involve relationships between entities

---

## THE 7 EDGE TYPES

Every edge MUST be one of these. Choosing the right type is critical.

| Type | Meaning | Example | Use When |
|------|---------|---------|----------|
| **MIRRORS** | A reflects/parallels B | "TCA MIRRORS Calculator" | Two things are structurally analogous |
| **INHERITS** | A derives from / depends on B | "App INHERITS Framework" | A couldn't exist without B |
| **BOUNDS** | A constrains / limits B | "FDA BOUNDS Pharma Companies" | A has power over B, sets rules |
| **EXPRESSES** | A produces / causes B | "Factory EXPRESSES Pollution" | A directly creates or leads to B |
| **VERIFIES** | A proves / grounds B | "Clinical Trial VERIFIES Drug Safety" | A provides evidence that B is real/true |
| **REMOVES** | A contradicts / destroys B | "Automation REMOVES Jobs" | A structurally undermines or kills B |
| **SEEKS** | A wants B but hasn't proven it | "Startup SEEKS Product-Market Fit" | The connection is aspirational, not verified |

### CRITICAL RULES:
- If a claim is **unproven**, use **SEEKS** not VERIFIES. Most plans are mostly SEEKS.
- **ALWAYS include REMOVES edges.** Every real system has contradictions. If your graph has zero REMOVES, you're being naive.
- **ALWAYS include BOUNDS edges.** Every real system has power structures and constraints.
- Don't make everything EXPRESSES. That's the lazy default. Think about what constrains, what contradicts, what's unproven.

---

## HOW TO BUILD THE GRAPH

You pass a JSON graph object to `tca_analyze_text`. The AI (you) builds the graph. TCA just analyzes.

### Format:
```json
{
  "name": "System Name",
  "nodes": [
    {"id": "snake_case_id", "label": "Human Readable Label"},
    {"id": "another_node", "label": "Another Node"}
  ],
  "edges": [
    {"source": "snake_case_id", "target": "another_node", "type": "EXPRESSES", "weight": 1.0}
  ]
}
```

### Rules:
- **10-25 nodes.** Fewer than 10 = too simple for TCA to find anything. More than 25 = noise.
- Node IDs must be `snake_case`. Labels can be anything human-readable.
- Every edge needs: `source` (node id), `target` (node id), `type` (one of the 7), `weight` (default 1.0).
- Include the **tensions**, not just the entities. If there's a power struggle, model both sides.

### Example Input (Small Startup):
```json
{
  "name": "Typical VC-Funded Startup",
  "nodes": [
    {"id": "founder", "label": "Founder"},
    {"id": "vc", "label": "VC Investors"},
    {"id": "product", "label": "Product"},
    {"id": "users", "label": "Users"},
    {"id": "revenue", "label": "Revenue"},
    {"id": "growth", "label": "Growth Metrics"},
    {"id": "burnrate", "label": "Burn Rate"},
    {"id": "mission", "label": "Original Mission"}
  ],
  "edges": [
    {"source": "founder", "target": "product", "type": "EXPRESSES", "weight": 1.0},
    {"source": "founder", "target": "mission", "type": "INHERITS", "weight": 1.0},
    {"source": "vc", "target": "founder", "type": "BOUNDS", "weight": 1.0},
    {"source": "vc", "target": "growth", "type": "SEEKS", "weight": 1.0},
    {"source": "product", "target": "users", "type": "SEEKS", "weight": 1.0},
    {"source": "users", "target": "revenue", "type": "SEEKS", "weight": 1.0},
    {"source": "revenue", "target": "burnrate", "type": "REMOVES", "weight": 1.0},
    {"source": "growth", "target": "mission", "type": "REMOVES", "weight": 1.0},
    {"source": "burnrate", "target": "founder", "type": "BOUNDS", "weight": 1.0}
  ]
}
```

Notice: VC BOUNDS Founder (power structure). Growth REMOVES Mission (contradiction). Users and Revenue are SEEKS (unproven). This is honest modeling.

---

## HOW TO READ TCA OUTPUT

TCA returns an `analysis` object containing:

- **node_count / edge_count**: Size of the graph
- **dead_ends**: Nodes that receive edges but send none. These are black holes — the system pours energy into them with no return. Ask: "Should this node produce something? What's it missing?"
- **orphan_sources**: Nodes that send edges but receive none. These appear from nowhere — no explanation for their existence. Ask: "Where does this come from? What justifies it?"
- **feedback_traps**: Bidirectional loops (A→B→A). These are cycles the system can't escape. Ask: "What external force could break this cycle?"
- **contradictions**: REMOVES edges. These are structural lies — things the system claims to want but actively undermines. These are often the most important findings.
- **star_nodes**: Nodes with disproportionate connections. Single points of failure. If this node dies, the system collapses.
- **solutions**: Proposed structural fixes, each with:
  - `problem_type`: What kind of problem (dead_end, orphan, trap, contradiction, star)
  - `problem_description`: What's wrong
  - `action`: What to do (add_edge, remove_edge, add_node)
  - `confidence`: How certain TCA is (0.0-1.0)
  - `reasoning`: Why this fix works topologically

### How to present results to the user:
1. Lead with the **contradictions** (REMOVES edges) — these are the insights that matter most
2. Then **dead ends** — where energy goes to die
3. Then **feedback traps** — where the system is stuck in loops
4. Then **solutions** — sorted by confidence, explain what each fix does structurally
5. Use the user's language, not graph terminology. "Revenue contradicts your open source goal" not "REMOVES edge detected between node revenue and node opensource"

---

## COMMON MISTAKES

❌ **Sending plain text instead of a graph JSON object.** TCA is a calculator, not a chatbot. It needs structured input.

❌ **Making every edge EXPRESSES.** This means you didn't think about power (BOUNDS), contradiction (REMOVES), or uncertainty (SEEKS). A graph with only EXPRESSES edges is useless.

❌ **No REMOVES edges.** Every real system has contradictions. If your graph is all harmony, you're lying.

❌ **Too few nodes (under 7).** TCA needs enough structure to find non-obvious patterns. A 3-node graph tells you nothing you didn't already know.

❌ **Modeling only the "official" structure.** Include the shadow structure — informal power, unspoken tensions, aspirational claims that aren't proven.

❌ **Using VERIFIES for unproven things.** If a startup "has product-market fit" but hasn't proven it with data, that's SEEKS, not VERIFIES. Be honest about what's real vs wished-for.

---

## THE 5 TCA TOOLS

1. **tca_analyze_text** — The main tool. You build the graph JSON, TCA analyzes it. Use for any custom system.
2. **tca_template** — Load a pre-built graph. Options: `economics`, `tanakh`, `us_geopolitics`, `china_geopolitics`, `apple`, `openai`. Good for demos or quick analysis.
3. **tca_solve** — Get solutions for an existing graph by graph_id. Returns fixes sorted by confidence.
4. **tca_apply** — Apply a specific solution by index. Modifies the graph and re-analyzes. Use to show "what if we fix this?"
5. **tca_export** — Export graph as `json` (full state) or `boot` (topology only, no labels — safe for sharing).

---

## WORKFLOW

Typical session:
1. User describes a system → You build a graph → Call `tca_analyze_text`
2. Present findings (contradictions first, then dead ends, then solutions)
3. User says "apply solution 2" → Call `tca_apply` with graph_id and index 2
4. Present updated analysis — show what changed
5. User wants to save it → Call `tca_export`

TCA is a calculator. You are the keyboard. The user is the one asking the question. Keep these roles clean.
