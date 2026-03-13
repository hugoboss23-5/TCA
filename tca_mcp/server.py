"""
TCA MCP Server — Topological reasoning as a native AI tool.

5 tools. One server. Any Claude instance gets structural analysis.
No API keys. No external calls. Pure topology.
"""

import json
import os

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
import uvicorn

from api import engine

server = Server("tca")


# --- Tool Definitions ---

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="tca_analyze_text",
            description=(
                "Analyze ANY system's structure. TCA finds structural problems "
                "(dead ends, feedback traps, star topologies, contradictions, "
                "ungrounded claims) with proposed solutions.\n\n"
                "YOU (the calling AI) must build the graph JSON. Read the user's "
                "description, identify entities and relationships, then pass a "
                "complete graph object.\n\n"
                "THE 7 EDGE TYPES (choose carefully — this is what makes TCA work):\n"
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
                "COMMON MISTAKES:\n"
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
                    "graph": {
                        "type": "object",
                        "description": (
                            "Complete graph object you build from the user's description. Format: "
                            "{\"name\": \"System Name\", "
                            "\"nodes\": [{\"id\": \"snake_case\", \"label\": \"Human Label\"}], "
                            "\"edges\": [{\"source\": \"id\", \"target\": \"id\", "
                            "\"type\": \"MIRRORS|INHERITS|BOUNDS|EXPRESSES|VERIFIES|REMOVES|SEEKS\", "
                            "\"weight\": 1.0}]}"
                        ),
                    },
                },
                "required": ["graph"],
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
        if "graph" not in arguments:
            return {"error": "Missing 'graph' parameter. Pass a JSON object with 'name', 'nodes', and 'edges'. See tool description for format."}
        return _analyze_graph(arguments["graph"])
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
