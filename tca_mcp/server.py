"""
TCA MCP Server — Topological reasoning as a native AI tool.

5 tools. One server. Any Claude instance gets structural analysis.
No API keys. No external calls. Pure topology.
"""

import json
import os
import re

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
import uvicorn

from api import engine

server = Server("tca")


# --- Text-to-Graph Parser (no AI, no API keys) ---

# Verbs/phrases that map to edge types.
_EDGE_PATTERNS = {
    "BOUNDS": [
        r"(?:controls?|constrains?|limits?|regulates?|governs?|manages?|oversees?|restricts?)",
        r"(?:reports?\s+to|accountable\s+to|answers?\s+to|under)",
        r"(?:has\s+(?:authority|power|control)\s+over)",
    ],
    "EXPRESSES": [
        r"(?:produces?|creates?|generates?|builds?|makes?|delivers?|outputs?|drives?|causes?)",
        r"(?:leads?\s+to|results?\s+in|feeds?|provides?|supplies?|enables?)",
    ],
    "REMOVES": [
        r"(?:contradicts?|conflicts?\s+with|opposes?|undermines?|blocks?|prevents?)",
        r"(?:competes?(?:\s+(?:with|for|against))?|competing)",
        r"(?:fights?\s+(?:with|against|over|for)|clashes?\s+with|tensions?\s+with)",
        r"(?:at\s+odds\s+with|incompatible\s+with)",
    ],
    "SEEKS": [
        r"(?:seeks?|wants?|needs?|tries?\s+to|attempts?\s+to|aims?\s+to|strives?\s+for)",
        r"(?:lacks?|missing|no\s+(?:direct\s+)?(?:access|connection|link|feedback|influence))",
        r"(?:cannot|can(?:'|no)t|unable\s+to|has\s+no)",
    ],
    "INHERITS": [
        r"(?:depends?\s+on|relies?\s+on|derives?\s+from|based\s+on|inherits?|requires?)",
        r"(?:part\s+of|belongs?\s+to|subset\s+of|branch\s+of)",
    ],
    "MIRRORS": [
        r"(?:similar\s+to|parallels?|resembles?|like|mirrors?|analogous\s+to)",
        r"(?:same\s+as|equivalent\s+to|corresponds?\s+to)",
    ],
    "VERIFIES": [
        r"(?:proves?|confirms?|validates?|verifies?|demonstrates?|evidence\s+(?:for|that))",
        r"(?:shows?\s+that|ensures?|guarantees?)",
    ],
}

# Compile patterns.
_COMPILED_EDGE_PATTERNS = {}
for _etype, _pats in _EDGE_PATTERNS.items():
    _COMPILED_EDGE_PATTERNS[_etype] = re.compile(
        "|".join(_pats), re.IGNORECASE
    )


def _parse_text_to_graph(description: str) -> dict:
    """Parse plain English into a TCA graph. No AI. Pure regex + heuristics.

    Strategy:
    1. Split into sentences.
    2. Extract noun phrases as candidate nodes.
    3. Detect relationship verbs to determine edge types.
    4. Build graph.
    """
    sentences = re.split(r'[.!?]+', description)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Extract entities: capitalized phrases, quoted terms, or noun-like chunks.
    entity_mentions = {}  # label -> id

    def _to_id(label: str) -> str:
        return re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')

    def _add_entity(label: str) -> str:
        label = label.strip()
        if not label or len(label) < 2:
            return ""
        # Strip leading articles.
        label = re.sub(r'^(?:the|a|an)\s+', '', label, flags=re.IGNORECASE).strip()
        if not label:
            return ""
        nid = _to_id(label)
        if not nid:
            return ""
        if nid not in entity_mentions:
            # Prefer capitalized version of the label.
            entity_mentions[nid] = label[0].upper() + label[1:] if label else label
        return nid

    edges = []

    _STOP_WORDS = frozenset([
        'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
        'has', 'have', 'had', 'this', 'that', 'it', 'they', 'he', 'she',
        'all', 'every', 'each', 'any', 'some', 'no', 'not', 'if', 'then',
        'when', 'while', 'for', 'with', 'from', 'to', 'in', 'on', 'at',
        'by', 'who', 'which', 'what', 'how', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall',
        'its', 'their', 'his', 'her', 'our', 'your', 'my', 'of', 'about',
        'also', 'just', 'only', 'very', 'much', 'more', 'most', 'other',
        'so', 'too', 'as', 'than', 'such', 'both', 'either', 'neither',
        'between', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
        'further', 'once', 'here', 'there', 'where', 'why', 'how', 'been',
        'being', 'having', 'doing', 'those', 'these', 'same', 'own',
        'see', 'sees', 'seen', 'get', 'gets', 'got', 'become', 'becomes',
        'problems', 'problem', 'issues', 'issue', 'things', 'thing',
        'way', 'ways', 'lot', 'lots', 'kind', 'type', 'part', 'parts',
        'company', 'organization', 'system', 'structure', 'process',
        'direct', 'directly', 'new', 'old', 'big', 'small', 'good', 'bad',
    ])

    # Junk phrases to reject.
    _JUNK_PATTERNS = re.compile(
        r'\b(with\s+a|but\s|sees?\s|problems?\b|priorities\b|feedback\s+loop)',
        re.IGNORECASE
    )

    def _is_valid_entity(phrase: str) -> bool:
        """Check if a phrase is a real entity, not a verb or junk."""
        words = phrase.lower().split()
        # All stop words = not an entity.
        if all(w in _STOP_WORDS for w in words):
            return False
        # Too short.
        if len(phrase) < 3:
            return False
        # Contains junk patterns.
        if _JUNK_PATTERNS.search(phrase):
            return False
        # Starts with a verb/preposition (common false positive).
        verb_starts = ('controls', 'manages', 'builds', 'sees', 'has',
                       'gets', 'does', 'makes', 'takes', 'gives', 'keeps')
        if words[0] in verb_starts:
            return False
        return True

    for sentence in sentences:
        # 1. Find capitalized multi-word phrases: "Customer Support", "CEO"
        caps = re.findall(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b', sentence)

        # 2. Find nouns that are objects of relationship verbs:
        #    "controls [all] [the] departments"
        verb_objects = re.findall(
            r'(?:controls?|manages?|builds?|produces?|influences?|affects?|'
            r'drives?|enables?|blocks?|prevents?|supports?|funds?|oversees?|'
            r'regulates?)\s+(?:all\s+)?(?:the\s+)?([A-Za-z]+)',
            sentence, re.IGNORECASE
        )

        # 3. Find "cannot influence X" patterns.
        negation_objects = re.findall(
            r'(?:cannot|can(?:\'|no)t)\s+'
            r'(?:influence|affect|reach|access|control|change)\s+'
            r'(?:the\s+)?([A-Z][A-Za-z]*(?:\s+[A-Za-z]+)?)',
            sentence
        )

        found_entities = []
        for phrase in caps + verb_objects + negation_objects:
            phrase = phrase.strip()
            if _is_valid_entity(phrase):
                nid = _add_entity(phrase)
                if nid:
                    found_entities.append(nid)

        # Deduplicate while preserving order.
        seen = set()
        unique_entities = []
        for e in found_entities:
            if e not in seen:
                seen.add(e)
                unique_entities.append(e)
        found_entities = unique_entities

        if len(found_entities) < 2:
            continue

        # Detect edge type from sentence.
        edge_type = "EXPRESSES"  # default
        for etype, pattern in _COMPILED_EDGE_PATTERNS.items():
            if pattern.search(sentence):
                edge_type = etype
                break

        # For "reports to" / "accountable to" — the direction is reversed
        # (A reports to B means B BOUNDS A).
        reverse = bool(re.search(
            r'(?:reports?\s+to|accountable\s+to|answers?\s+to|under\b)',
            sentence, re.IGNORECASE
        ))

        # Create edges between consecutive entity pairs in the sentence.
        for i in range(len(found_entities) - 1):
            src, tgt = found_entities[i], found_entities[i + 1]
            if src == tgt:
                continue
            if reverse and edge_type == "BOUNDS":
                src, tgt = tgt, src
            edges.append({
                "source": src,
                "target": tgt,
                "type": edge_type,
                "weight": 1.0,
            })

    # Build node list.
    nodes = [{"id": nid, "label": label} for nid, label in entity_mentions.items()]

    # Deduplicate edges.
    seen_edges = set()
    unique_edges = []
    for e in edges:
        key = (e["source"], e["target"], e["type"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)

    # Derive a name from the first few entities.
    name = "System"
    if nodes:
        name = " / ".join(n["label"] for n in nodes[:3])

    return {
        "name": name,
        "description": description,
        "nodes": nodes,
        "edges": unique_edges,
    }


# --- Tool Definitions ---

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="tca_analyze_text",
            description=(
                "Describe ANY system in plain English. TCA maps it as a "
                "topological graph and finds structural problems (dead ends, "
                "feedback traps, star topologies, contradictions, ungrounded "
                "claims) with proposed solutions. Use this whenever someone "
                "wants to understand the structure of any system — a company, "
                "a country, a religion, an economy, a relationship, a plan.\n\n"
                "Pass a plain text description. TCA extracts entities and "
                "relationships automatically, builds the graph, and analyzes it.\n\n"
                "Include: key entities, who controls whom, what produces what, "
                "what contradicts what, what's unproven. More detail = better graph."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": (
                            "Plain English description of the system to analyze. "
                            "Name the entities, describe relationships, mention "
                            "tensions and goals."
                        ),
                    },
                },
                "required": ["description"],
            },
        ),
        Tool(
            name="tca_template",
            description=(
                "Load a pre-built TCA graph with instant analysis. "
                "Templates: economics, tanakh, us_geopolitics, "
                "china_geopolitics, apple, openai."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "template_name": {
                        "type": "string",
                        "enum": [
                            "economics", "tanakh", "us_geopolitics",
                            "china_geopolitics", "apple", "openai",
                        ],
                    }
                },
                "required": ["template_name"],
            },
        ),
        Tool(
            name="tca_solve",
            description=(
                "Get TCA solutions for an existing graph. Returns proposed "
                "structural fixes sorted by confidence."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_id": {"type": "string"}
                },
                "required": ["graph_id"],
            },
        ),
        Tool(
            name="tca_apply",
            description=(
                "Apply a TCA solution to a graph by index. Modifies the "
                "graph and returns updated analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_id": {"type": "string"},
                    "solution_index": {"type": "integer"},
                },
                "required": ["graph_id", "solution_index"],
            },
        ),
        Tool(
            name="tca_export",
            description=(
                "Export a TCA graph. Format 'json' = full state, "
                "'boot' = architecture only (no private data, safe for "
                "open source)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_id": {"type": "string"},
                    "format": {
                        "type": "string",
                        "enum": ["json", "boot"],
                    },
                },
                "required": ["graph_id", "format"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        result = _handle_tool(name, arguments)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str),
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)}),
        )]


def _handle_tool(name: str, arguments: dict) -> dict:
    if name == "tca_analyze_text":
        if "description" not in arguments:
            return {"error": "Missing 'description' parameter. Pass a plain text description of the system to analyze."}
        return _analyze_text(arguments["description"])
    elif name == "tca_template":
        return _load_template(arguments["template_name"])
    elif name == "tca_solve":
        return _solve(arguments["graph_id"])
    elif name == "tca_apply":
        return _apply(arguments["graph_id"], arguments["solution_index"])
    elif name == "tca_export":
        return _export(arguments["graph_id"], arguments["format"])
    else:
        return {"error": f"Unknown tool: {name}"}


def _analyze_text(description: str) -> dict:
    """Text -> Graph -> Analysis. Parses plain English into a graph, then analyzes."""
    graph_data = _parse_text_to_graph(description)

    if not graph_data.get("nodes"):
        return {
            "error": (
                "Could not extract entities from the description. "
                "Try naming specific entities with capital letters: "
                "'The CEO controls Marketing. Sales competes with Marketing for budget.'"
            )
        }

    return _analyze_graph(graph_data)


def _analyze_graph(graph_data: dict) -> dict:
    """Graph -> Analysis. Loads graph data into TCA engine and runs analysis."""
    result = engine.create_graph(name=graph_data.get("name", "Auto"))
    graph_id = result["graph_id"]

    for node in graph_data.get("nodes", []):
        engine.add_node(graph_id, node["label"], node["id"])

    for edge in graph_data.get("edges", []):
        try:
            engine.add_edge(
                graph_id,
                edge["source"],
                edge["target"],
                edge["type"],
                edge.get("weight", 1.0),
            )
        except (ValueError, KeyError):
            continue

    analysis = engine.run_analysis(graph_id)

    return {
        "graph_id": graph_id,
        "graph": graph_data,
        "analysis": analysis,
    }


def _load_template(template_name: str) -> dict:
    result = engine.create_graph(
        name=template_name.replace("_", " ").title(),
        template=template_name,
    )
    graph_id = result["graph_id"]
    analysis = engine.run_analysis(graph_id)
    return {
        "graph_id": graph_id,
        "template": template_name,
        "analysis": analysis,
    }


def _solve(graph_id: str) -> dict:
    analysis = engine.run_analysis(graph_id)
    if analysis is None:
        return {"error": f"Graph '{graph_id}' not found"}
    solutions = analysis.get("solutions", [])
    return {"solutions": solutions, "total": len(solutions)}


def _apply(graph_id: str, index: int) -> dict:
    result = engine.apply_solution(graph_id, index)
    if result is None:
        return {"error": f"Graph '{graph_id}' not found"}
    return result


def _export(graph_id: str, fmt: str) -> dict:
    if fmt == "boot":
        return engine.export_boot(graph_id) or {"error": "Graph not found"}
    return engine.export_state(graph_id) or {"error": "Graph not found"}


# --- SSE Transport ---

from fastapi import FastAPI, Request
from starlette.responses import Response
from starlette.routing import Route

sse = SseServerTransport("/messages/")


async def handle_sse_endpoint(request: Request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0], streams[1], server.create_initialization_options()
        )
    return Response()


async def handle_messages_endpoint(request: Request):
    await sse.handle_post_message(request.scope, request.receive, request._send)


routes = [
    Route("/sse", handle_sse_endpoint),
    Route("/messages/", handle_messages_endpoint, methods=["POST"]),
]

app = FastAPI(routes=routes)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8100))
    uvicorn.run(app, host="0.0.0.0", port=port)
