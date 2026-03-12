"""
KD Vault → local JSON loader.

Pulls ALL nodes from the KD vault via mcp__KD__vault_browse (MCP tool),
extracts id, content, domain, confidence, tags from each node,
and writes the full result to tca/data/all_nodes.json.

Execution method:
    This is NOT a standalone Python script. The data was pulled by the
    Claude Code agent using mcp__KD__vault_browse with pagination
    (limit=50, offset incremented by 50 each batch), then assembled
    from the conversation transcript and persisted tool results.

Pagination strategy:
    - facts:   19 batches × 50 = 930 nodes  (offsets 0–900)
    - beliefs: 10 batches × 50 = 500 nodes  (offsets 0–450)
    - Total:   1,430 nodes

Output:
    tca/data/all_nodes.json — JSON array of objects with keys:
        id, content, domain, confidence, tags

Generated: 2026-03-12
"""
