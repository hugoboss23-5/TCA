"""
TCA MCP Server — Topological reasoning as a native AI tool.

5 tools. One server. Any Claude instance gets structural analysis.
"""

import json
import os

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
import uvicorn

from api import engine

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

server = Server("tca")


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
                "TWO MODES:\n"
                "1. Pass 'description' (plain text) — TCA auto-builds the graph via AI, then analyzes it.\n"
                "2. Pass 'graph' (JSON object) — you build the graph yourself for maximum control.\n\n"
                "THE 7 EDGE TYPES (for mode 2, or to understand output):\n"
                "- MIRRORS: A parallels B (analogy). 'TCA MIRRORS Calculator'\n"
                "- INHERITS: A depends on B. 'App INHERITS Framework'\n"
                "- BOUNDS: A constrains/controls B (power). 'FDA BOUNDS Pharma'\n"
                "- EXPRESSES: A produces/causes B. 'Factory EXPRESSES Pollution'\n"
                "- VERIFIES: A proves B is real. 'Trial VERIFIES Drug Safety'\n"
                "- REMOVES: A contradicts/destroys B. 'Automation REMOVES Jobs'\n"
                "- SEEKS: A wants B but unproven. 'Startup SEEKS Product-Market Fit'\n\n"
                "CRITICAL RULES:\n"
                "- Use SEEKS (not VERIFIES) for unproven claims. Most plans are mostly SEEKS.\n"
                "- ALWAYS include REMOVES edges. Every system has contradictions. Zero REMOVES = naive graph.\n"
                "- ALWAYS include BOUNDS edges. Every system has power structures.\n"
                "- Don't make everything EXPRESSES. Think about what constrains, contradicts, is unproven.\n"
                "- 10-25 nodes. Under 10 = too simple. Over 25 = noise.\n\n"
                "COMMON MISTAKES (when building graph yourself):\n"
                "- All EXPRESSES edges → useless analysis\n"
                "- No REMOVES → you're hiding contradictions\n"
                "- Under 7 nodes → nothing non-obvious to find\n"
                "- Using VERIFIES for hopes → use SEEKS\n\n"
                "HOW TO READ OUTPUT:\n"
                "- Lead with contradictions (REMOVES) — most valuable insight\n"
                "- Then dead ends — where energy goes to die\n"
                "- Then feedback traps — stuck loops\n"
                "- Then solutions sorted by confidence\n"
                "- Translate graph terms to user's language\n\n"
                "WHEN TO USE: 'analyze structure of', 'what's broken about', "
                "'contradictions in', 'run TCA on', any system/org/plan analysis.\n"
                "WHEN NOT TO USE: simple facts, coding, math, creative writing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": (
                            "Plain English description of the system. "
                            "TCA will auto-build the graph via AI. "
                            "Include key entities, relationships, tensions, goals. "
                            "More detail = better graph."
                        ),
                    },
                    "graph": {
                        "type": "object",
                        "description": (
                            "OR: build the graph yourself for maximum control. Format: "
                            "{\"name\": \"System Name\", "
                            "\"nodes\": [{\"id\": \"snake_case\", \"label\": \"Human Label\"}], "
                            "\"edges\": [{\"source\": \"id\", \"target\": \"id\", "
                            "\"type\": \"MIRRORS|INHERITS|BOUNDS|EXPRESSES|VERIFIES|REMOVES|SEEKS\", "
                            "\"weight\": 1.0}]}"
                        ),
                    },
                },
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
        result = await _handle_tool(name, arguments)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str),
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)}),
        )]


async def _handle_tool(name: str, arguments: dict) -> dict:
    if name == "tca_analyze_text":
        if "graph" in arguments:
            return _analyze_graph(arguments["graph"])
        elif "description" in arguments:
            return await _analyze_text(arguments["description"])
        else:
            return {"error": "Pass either 'description' (plain text for auto-graph) or 'graph' (JSON object you build yourself). See tool description."}
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


def _analyze_graph(graph_data: dict) -> dict:
    """Graph -> Analysis. The calling AI builds the graph, TCA analyzes it."""
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


# --- Text-to-graph system prompt ---

_GRAPH_SYSTEM_PROMPT = """You are a topology builder. Convert the description into a JSON graph.
Return ONLY valid JSON. No markdown. No explanation.

Edge types: MIRRORS (analogy), INHERITS (derives from), BOUNDS (constrains), EXPRESSES (produces/causes), VERIFIES (proves), REMOVES (contradicts/destroys), SEEKS (wants but unproven).

Rules: unproven claims use SEEKS not VERIFIES. ALWAYS include contradictions (REMOVES). Include power structures (BOUNDS). 10-25 nodes.

JSON: {"name":"...","description":"...","nodes":[{"id":"snake_case","label":"Label"}],"edges":[{"source":"id","target":"id","type":"TYPE","weight":1.0}]}"""


async def _analyze_text(description: str) -> dict:
    """Text -> Graph -> Analysis. Auto-builds graph from plain English."""
    if not ANTHROPIC_API_KEY:
        return {
            "error": (
                "ANTHROPIC_API_KEY not set. Export it as an environment "
                "variable. Alternatively, pass 'graph' instead of "
                "'description' to build the graph yourself without needing an API key."
            )
        }

    graph_data = await _build_graph_from_text(description)

    if graph_data is None:
        return {
            "error": (
                "Failed to generate graph from text. Try rephrasing with more "
                "specific entities and relationships, or pass a 'graph' object directly."
            )
        }

    # Reuse the graph analysis path.
    result = _analyze_graph(graph_data)
    result["description"] = description
    return result


async def _build_graph_from_text(description: str) -> dict | None:
    """Convert plain text to graph JSON via Anthropic API. Async."""
    import httpx

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4096,
                "system": _GRAPH_SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Build a TCA graph for:\n\n{description}",
                    }
                ],
            },
        )

    if r.status_code != 200:
        return None

    text = "".join(
        b.get("text", "")
        for b in r.json().get("content", [])
        if b.get("type") == "text"
    )
    text = text.strip().strip("`").strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


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


# --- SSE Transport (matching KD's proven pattern) ---

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
