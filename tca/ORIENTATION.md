# TCA Orientation Report
## Filesystem & System Mapping — March 2026

---

## Repository State

- **Repo:** /home/user/TCA (empty repo, no pre-existing files)
- **Branch:** claude/build-tca-system-qlCMh
- **No local code** for Saturn, Jarvis, Watty, Basho, FEELD, or Sunlight exists in this repository.
- Existing systems are accessible exclusively via **MCP (Model Context Protocol) servers**.

---

## System 1: KD (Knowledge Dynamics) — MCP Server

**Language/Framework:** MCP server (accessed via tool calls, not direct code import)
**Purpose:** Knowledge storage, retrieval, memory decay, graph reasoning

### Core Data Structures
- **Facts:** High-confidence knowledge items (default confidence 1.0)
  - Ontology types: claim, definition, measurement, event, source
- **Beliefs:** Lower-confidence knowledge items (default confidence 0.5)
  - Ontology types: hypothesis, principle, prediction, preference, pattern
  - FSRS memory state: stability, difficulty, retrievability, due date
- **Edges:** Graph connections between knowledge items (trigram similarity, shared domain, tag overlap)
- **ACH Matrices:** Analysis of Competing Hypotheses with evidence ratings

### API Surface (MCP Tools)

#### Vault Operations
| Tool | Function |
|------|----------|
| `vault_deposit` | Store fact or belief with confidence, domain, tags, reliability grade (A-F) |
| `vault_query` | Keyword search ranked by relevance x confidence |
| `vault_browse` | Paginated browsing with sort (recent/confidence/alphabetical) |
| `vault_delete` | Archive and remove a fact/belief by UUID |
| `vault_protect` | Mark as too-big-to-fail (protected=TRUE) |
| `vault_stats` | Totals, avg confidence, daily counts, reserve ratio |

#### Graph Operations
| Tool | Function |
|------|----------|
| `graph_build` | Build/rebuild edges (trigram similarity threshold, default 0.25) |
| `graph_query` | **Spreading activation search** — keyword match then follow edges. Params: query, depth (max 3), decay (0-1, default 0.6), limit |
| `graph_stats` | Total edges, edge types, most connected nodes |

#### Memory & Decay (FSRS)
| Tool | Function |
|------|----------|
| `belief_review` | Flashcard-style review: again/hard/good/easy. Updates FSRS state |
| `belief_health` | Show FSRS state: stability, difficulty, retrievability, due date |
| `run_decay` | Run decay cycle, archive beliefs below death threshold |

#### Analysis
| Tool | Function |
|------|----------|
| `ach_create` | Create ACH matrix with question + 2-7 hypotheses |
| `ach_evidence` | Add evidence with ratings (II/I/N/C/CC) per hypothesis |
| `ach_evaluate` | Score hypotheses using CIA inconsistency-weighted method |
| `ach_list` | List all ACH matrices |
| `contradiction_scan` | Find contradicting belief pairs by domain |

#### System
| Tool | Function |
|------|----------|
| `kd_health` | Full system health check |
| `fed_policy` | Current system health with strongest/weakest beliefs |
| `fed_audit` | Identify problems: risk-flagged, contradicted, duplicates |
| `fed_inject` | Force beliefs into next agent's context |
| `metabolize` | Batch deposit: facts, beliefs, reinforcements, contradictions, reasoning chains |
| `record_outcome` | Close the loop: confirmed/refuted/partially_confirmed/superseded/expired |
| `get_context_packet` | Boot context for agents (top beliefs, facts, outcomes, active state) |
| `set_active_focus` | Announce what this agent is working on |
| `clear_active_focus` | Mark work completed |
| `convictions` | All beliefs with confidence > 0.8 |
| `sitrep` | Daily briefing |
| `ping` | Health check |
| `list_domains` | All unique knowledge domains with counts |

### Key Observations for TCA Integration
- KD already has **spreading activation** (`graph_query` with depth/decay params) — TCA L2 adapter can leverage this
- KD uses **FSRS memory decay** — TCA L4 temporal engine should respect this
- KD's graph is built on **trigram similarity + domain + tags** — TCA L2 will add topological edge types (MIRRORS, INHERITS, etc.)
- KD has **outcome recording** — natural fit for TCA L3 grounding loop

---

## System 2: Stork — MCP Server (Agent Orchestrator)

**Language/Framework:** MCP server (Python backend based on parameter style)
**Purpose:** Multi-agent orchestration with lane-based scheduling

### Core Concepts
- **Lanes:** now (immediate), tonight (batch), campaign (long-running), watch (monitoring)
- **Profiles:** Agent configurations (quant, research, code, fast)
- **Campaigns:** Goal-tracked multi-step workflows

### API Surface (MCP Tools)

| Tool | Function |
|------|----------|
| `spawn` | Single agent, NOW lane. Optional profile and timeout |
| `spawn_many` | Parallel agents, NOW lane. Per-agent status reporting |
| `chain` | Sequential pipeline, NOW lane. Shows all steps including failures |
| `delegate` | Plan -> parallel execute -> synthesize, NOW lane |
| `campaign_create` | Create campaign with goal and lane |
| `campaign_status` | Status of one or all campaigns |
| `campaign_redirect` | Pivot a campaign with new instruction |
| `campaign_kill` | Stop campaign, write post-mortem |
| `overnight_start` | Begin processing TONIGHT + CAMPAIGN lanes |
| `overnight_stop` | Graceful shutdown with sit-rep |
| `profiles` | List available agent profiles |
| `status` | All running agents + campaign overview |
| `convictions` | Current beliefs as natural language |
| `sitrep` | Daily briefing (defaults to today) |

### Key Observations for TCA Integration
- Stork is an **orchestration layer** — TCA does not directly depend on it
- Stork could be used to run TCA reasoning loops as spawned agents in Phase 2
- No direct adapter needed for v0.1

---

## Systems NOT Present

| System | Status |
|--------|--------|
| Saturn | Not in this repo. No MCP tools found. Cannot map. |
| Jarvis (Chestohedron Router) | Not in this repo. No MCP tools found. Cannot map. |
| Watty/Basho | Not in this repo. No MCP tools found. Cannot map. |
| FEELD | Not in this repo. No MCP tools found. Cannot map. |
| Sunlight | Not in this repo. No MCP tools found. Cannot map. |

---

## Architecture Decision: Adapters

Since KD and Stork are MCP servers (not importable Python modules), adapters will:
1. **KD adapter:** Wrap MCP tool calls into Python functions that TCA layers can call
2. **Jarvis adapter:** Since Jarvis is not available, L1 router will implement its own 7-gate routing based on the Chestohedron doctrine (all gates activate simultaneously)
3. **Stork adapter:** Not needed for v0.1

---

## TCA Directory Structure

```
tca/
  ORIENTATION.md          <- This file (ground truth)
  adapters/               <- Interfaces to existing systems
    kd_adapter.py         <- Wraps KD MCP tools
    jarvis_adapter.py     <- Stub (Jarvis not available)
  L1_router/              <- Chestohedron 7-gate router
  L2_graph/               <- Topological knowledge graph
  L3_grounding/           <- Prediction-error loops
  L4_temporal/            <- Internal clock + reasoning
  L5_metacognition/       <- Self-monitoring
  tests/                  <- All test files
```
