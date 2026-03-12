/* TCA Calculator — React + D3 Frontend (V1 single file) */

const { useState, useEffect, useRef, useCallback } = React;

// ── Edge type colors ───────────────────────────────────────────
const EDGE_COLORS = {
  MIRRORS: '#4fc3f7', INHERITS: '#81c784', BOUNDS: '#ff8a65',
  EXPRESSES: '#ba68c8', VERIFIES: '#fff176', REMOVES: '#ef5350',
  SEEKS: '#90a4ae',
};

const EDGE_TOOLTIPS = {
  MIRRORS: 'Analogous to — reflects or parallels',
  INHERITS: 'Derives from — lineage or origin',
  BOUNDS: 'Constrains — sets limits',
  EXPRESSES: 'Manifests as — produces output',
  VERIFIES: 'Provides evidence for — proves',
  REMOVES: 'Contradicts — this is a contradiction',
  SEEKS: 'Unproven connection — TCA will ask for evidence',
};

const EDGE_TYPES = Object.keys(EDGE_COLORS);

// ── API helper ─────────────────────────────────────────────────
const api = {
  async createGraph(name, template) {
    const r = await fetch('/api/graphs', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ name, template }),
    });
    return r.json();
  },
  async getGraph(id) { return (await fetch(`/api/graphs/${id}`)).json(); },
  async listGraphs() { return (await fetch('/api/graphs')).json(); },
  async deleteGraph(id) { return fetch(`/api/graphs/${id}`, { method: 'DELETE' }); },
  async addNode(gid, label, nodeId) {
    const r = await fetch(`/api/graphs/${gid}/nodes`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ label, node_id: nodeId || undefined }),
    });
    return r.json();
  },
  async deleteNode(gid, nid) {
    return fetch(`/api/graphs/${gid}/nodes/${nid}`, { method: 'DELETE' });
  },
  async addEdge(gid, sourceId, targetId, edgeType, weight) {
    const r = await fetch(`/api/graphs/${gid}/edges`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ source_id: sourceId, target_id: targetId, edge_type: edgeType, weight }),
    });
    return r.json();
  },
  async analyze(gid) {
    const r = await fetch(`/api/graphs/${gid}/analyze`, { method: 'POST' });
    return r.json();
  },
  async applySolution(gid, idx) {
    const r = await fetch(`/api/graphs/${gid}/solutions/${idx}/apply`, { method: 'POST' });
    return r.json();
  },
  async getTemplates() { return (await fetch('/api/templates')).json(); },
  async exportState(gid) { return (await fetch(`/api/graphs/${gid}/export/state`)).json(); },
  async exportBoot(gid) { return (await fetch(`/api/graphs/${gid}/export/boot`)).json(); },
};

// ── Graph Canvas (D3 force-directed) ──────────────────────────
function GraphCanvas({ graphState, analysis, selectedNode, onSelectNode }) {
  const svgRef = useRef(null);
  const simRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !graphState) return;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth || 800;
    const height = svgRef.current.clientHeight || 500;

    svg.selectAll('*').remove();

    const nodes = graphState.nodes.map(n => ({ ...n }));
    const links = [];
    graphState.nodes.forEach(n => {
      n.edges.forEach(e => {
        links.push({ source: n.id, target: e.target_id, edge_type: e.edge_type, weight: e.weight });
      });
    });

    if (nodes.length === 0) {
      svg.append('text').attr('x', width/2).attr('y', height/2)
        .attr('text-anchor', 'middle').attr('fill', '#888').attr('font-size', 14)
        .text('Add nodes or load a template to begin');
      return;
    }

    // Problem sets for highlighting
    const deadEnds = new Set();
    const orphans = new Set();
    const starCenters = new Set();
    const trapNodes = new Set();
    if (analysis && analysis.solutions) {
      analysis.solutions.forEach(s => {
        if (s.problem_type === 'dead_end') deadEnds.add(s.details.node_id);
        if (s.problem_type === 'orphan') orphans.add(s.details.node_id);
        if (s.problem_type === 'star') starCenters.add(s.details.center_id);
        if (s.problem_type === 'trap') {
          trapNodes.add(s.details.loop_node_a);
          trapNodes.add(s.details.loop_node_b);
        }
      });
    }

    const g = svg.append('g');

    // Zoom
    svg.call(d3.zoom().scaleExtent([0.3, 3]).on('zoom', (event) => {
      g.attr('transform', event.transform);
    }));

    // Arrow markers
    EDGE_TYPES.forEach(et => {
      svg.append('defs').append('marker')
        .attr('id', `arrow-${et}`).attr('viewBox', '0 -5 10 10')
        .attr('refX', 22).attr('refY', 0).attr('markerWidth', 6).attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', EDGE_COLORS[et]);
    });

    // Links
    const link = g.append('g').selectAll('line').data(links).join('line')
      .attr('class', 'edge-line')
      .attr('stroke', d => EDGE_COLORS[d.edge_type] || '#555')
      .attr('stroke-dasharray', d => (d.edge_type === 'SEEKS' || d.edge_type === 'REMOVES') ? '6,3' : null)
      .attr('marker-end', d => `url(#arrow-${d.edge_type})`);

    // Edge labels
    const edgeLabel = g.append('g').selectAll('text').data(links).join('text')
      .attr('class', 'edge-label')
      .text(d => d.edge_type);

    // Nodes
    const node = g.append('g').selectAll('g').data(nodes).join('g')
      .call(d3.drag()
        .on('start', (event, d) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on('end', (event, d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    node.append('circle')
      .attr('class', d => {
        let cls = 'node-circle';
        if (deadEnds.has(d.id)) cls += ' dead-end';
        if (orphans.has(d.id)) cls += ' orphan';
        if (starCenters.has(d.id)) cls += ' star-center';
        if (trapNodes.has(d.id)) cls += ' in-trap';
        if (selectedNode === d.id) cls += ' selected';
        return cls;
      })
      .attr('r', d => 8 + Math.min(d.edges.length * 2, 12))
      .attr('fill', d => {
        if (d.edges.length === 0) return '#444';
        const types = {};
        d.edges.forEach(e => { types[e.edge_type] = (types[e.edge_type] || 0) + 1; });
        const dominant = Object.keys(types).sort((a,b) => types[b] - types[a])[0];
        return EDGE_COLORS[dominant] || '#888';
      })
      .attr('fill-opacity', 0.25)
      .attr('stroke', d => {
        if (d.edges.length === 0) return '#666';
        const types = {};
        d.edges.forEach(e => { types[e.edge_type] = (types[e.edge_type] || 0) + 1; });
        const dominant = Object.keys(types).sort((a,b) => types[b] - types[a])[0];
        return EDGE_COLORS[dominant] || '#888';
      })
      .on('click', (event, d) => { event.stopPropagation(); onSelectNode(d.id); });

    node.append('text').attr('class', 'node-label').attr('dy', d => -(10 + Math.min(d.edges.length * 2, 12)))
      .text(d => d.label.length > 20 ? d.label.slice(0, 18) + '...' : d.label);

    // Simulation
    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(120).strength(0.5))
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    sim.on('tick', () => {
      link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      edgeLabel.attr('x', d => (d.source.x + d.target.x) / 2)
               .attr('y', d => (d.source.y + d.target.y) / 2 - 4);
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    simRef.current = sim;

    // Click canvas to deselect
    svg.on('click', () => onSelectNode(null));

    return () => { sim.stop(); };
  }, [graphState, analysis, selectedNode]);

  return (
    <div className="canvas-container">
      <svg ref={svgRef}></svg>
    </div>
  );
}

// ── Sidebar ───────────────────────────────────────────────────
function Sidebar({ graphId, graphState, selectedNode, onSelectNode, onRefresh }) {
  const [nodeLabel, setNodeLabel] = useState('');
  const [edgeSource, setEdgeSource] = useState('');
  const [edgeTarget, setEdgeTarget] = useState('');
  const [edgeType, setEdgeType] = useState('MIRRORS');
  const [edgeWeight, setEdgeWeight] = useState(1.0);

  const nodes = graphState ? graphState.nodes : [];

  const handleAddNode = async () => {
    if (!nodeLabel.trim() || !graphId) return;
    await api.addNode(graphId, nodeLabel.trim());
    setNodeLabel('');
    onRefresh();
  };

  const handleAddEdge = async () => {
    if (!edgeSource || !edgeTarget || !graphId || edgeSource === edgeTarget) return;
    await api.addEdge(graphId, edgeSource, edgeTarget, edgeType, edgeWeight);
    onRefresh();
  };

  const handleDeleteNode = async (nid) => {
    if (!graphId) return;
    await api.deleteNode(graphId, nid);
    if (selectedNode === nid) onSelectNode(null);
    onRefresh();
  };

  return (
    <div className="sidebar">
      <section>
        <h3>Add Node</h3>
        <input placeholder="Node label..." value={nodeLabel}
          onChange={e => setNodeLabel(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleAddNode()} />
        <button onClick={handleAddNode}>Add Node</button>
      </section>

      <section>
        <h3>Add Edge</h3>
        <select value={edgeSource} onChange={e => setEdgeSource(e.target.value)}>
          <option value="">Source...</option>
          {nodes.map(n => <option key={n.id} value={n.id}>{n.label}</option>)}
        </select>
        <select value={edgeTarget} onChange={e => setEdgeTarget(e.target.value)}>
          <option value="">Target...</option>
          {nodes.map(n => <option key={n.id} value={n.id}>{n.label}</option>)}
        </select>
        <select value={edgeType} onChange={e => setEdgeType(e.target.value)}
          title={EDGE_TOOLTIPS[edgeType]}>
          {EDGE_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
        <button onClick={handleAddEdge} title={EDGE_TOOLTIPS[edgeType]}>
          Add Edge ({edgeType})
        </button>
      </section>

      <section>
        <h3>Nodes ({nodes.length})</h3>
        <div className="node-list">
          {nodes.map(n => (
            <div key={n.id}
              className={`node-item${selectedNode === n.id ? ' selected' : ''}`}
              onClick={() => onSelectNode(n.id)}>
              <span>{n.label} <span style={{color:'#666'}}>({n.edges.length})</span></span>
              <button className="delete-btn" onClick={e => { e.stopPropagation(); handleDeleteNode(n.id); }}
                title="Delete node">x</button>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

// ── Analysis Panel ────────────────────────────────────────────
function AnalysisPanel({ graphId, analysis, onRefresh }) {
  if (!analysis) {
    return (
      <div className="analysis-panel">
        <div className="analysis-section"><div className="empty-state">No analysis yet. Add nodes and edges, then analysis runs automatically.</div></div>
      </div>
    );
  }

  const handleApply = async (idx) => {
    if (!graphId) return;
    await api.applySolution(graphId, idx);
    onRefresh();
  };

  const health = analysis.health;
  const conf = analysis.confidence;

  return (
    <div className="analysis-panel">
      {/* Health */}
      <div className="analysis-section">
        <h3>Topology Health</h3>
        <div className="metric-item"><span>Nodes</span><span className="value">{analysis.node_count}</span></div>
        <div className="metric-item"><span>Edges</span><span className="value">{analysis.edge_count}</span></div>
        <div className="metric-item"><span>Cycles</span><span className="value">{health.cycles.length}</span></div>
        <div className="metric-item"><span>Bridges</span><span className="value">{health.bridges.length}</span></div>
        <div className="metric-item"><span>Isolated</span><span className="value">{health.isolated.length}</span></div>
        <div className="metric-item"><span>Confidence</span><span className="value">{(conf.confidence * 100).toFixed(1)}%</span></div>
        <div className="metric-item"><span>Grounding</span><span className="value">{(conf.grounding_ratio * 100).toFixed(1)}%</span></div>
      </div>

      {/* Solutions */}
      <div className="analysis-section" style={{flex: 2}}>
        <h3>Solutions ({analysis.solution_count})</h3>
        {analysis.solutions.slice(0, 20).map(s => (
          <div key={s.index} className="solution-card">
            <div className="problem-type" style={{color: EDGE_COLORS[
              s.problem_type === 'dead_end' ? 'REMOVES' :
              s.problem_type === 'orphan' ? 'VERIFIES' :
              s.problem_type === 'star' ? 'BOUNDS' :
              s.problem_type === 'trap' ? 'SEEKS' :
              s.problem_type === 'ungrounded' ? 'SEEKS' : 'REMOVES'
            ]}}>
              {s.problem_type}
            </div>
            <div className="description">{s.problem_description}</div>
            <div className="reasoning">{s.reasoning}</div>
            <div className="confidence-bar">
              <div className="confidence-fill" style={{width: `${s.confidence * 100}%`}}></div>
            </div>
            <button onClick={() => handleApply(s.index)}>Apply ({(s.confidence * 100).toFixed(0)}%)</button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Top Bar ───────────────────────────────────────────────────
function TopBar({ templates, graphId, graphName, onLoadTemplate, onExport }) {
  const [tpl, setTpl] = useState('');

  const handleLoad = () => {
    if (tpl) onLoadTemplate(tpl);
  };

  return (
    <div className="topbar">
      <h1>TCA</h1>
      <select value={tpl} onChange={e => setTpl(e.target.value)}>
        <option value="">Load template...</option>
        {templates.map(t => (
          <option key={t.name} value={t.name}>{t.name} ({t.node_count} nodes)</option>
        ))}
      </select>
      <button className="primary" onClick={handleLoad} disabled={!tpl}>Load</button>
      <div className="spacer"></div>
      {graphName && <span className="graph-name">{graphName}</span>}
      <button onClick={() => onExport('state')} disabled={!graphId}>Export JSON</button>
      <button onClick={() => onExport('boot')} disabled={!graphId}>Export Boot</button>
      <button disabled title="Coming soon — publish your graph to the ecosystem">Publish</button>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────
function App() {
  const [graphId, setGraphId] = useState(null);
  const [graphName, setGraphName] = useState('');
  const [graphState, setGraphState] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [templates, setTemplates] = useState([]);

  // Load templates on mount
  useEffect(() => {
    api.getTemplates().then(data => setTemplates(data.templates || []));
  }, []);

  // Create a blank graph on mount
  useEffect(() => {
    api.createGraph('Untitled').then(data => {
      setGraphId(data.graph_id);
      setGraphName('Untitled');
      refreshGraph(data.graph_id);
    });
  }, []);

  const refreshGraph = useCallback(async (gid) => {
    const id = gid || graphId;
    if (!id) return;
    const state = await api.getGraph(id);
    setGraphState(state);
    // Auto-analyze if graph has nodes
    if (state.node_count > 0) {
      const a = await api.analyze(id);
      setAnalysis(a);
    } else {
      setAnalysis(null);
    }
  }, [graphId]);

  const handleLoadTemplate = async (templateName) => {
    const data = await api.createGraph(templateName, templateName);
    setGraphId(data.graph_id);
    setGraphName(templateName);
    setSelectedNode(null);
    refreshGraph(data.graph_id);
  };

  const handleExport = async (format) => {
    if (!graphId) return;
    const data = format === 'boot'
      ? await api.exportBoot(graphId)
      : await api.exportState(graphId);
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tca-${graphName || 'export'}-${format}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <TopBar
        templates={templates}
        graphId={graphId}
        graphName={graphName}
        onLoadTemplate={handleLoadTemplate}
        onExport={handleExport}
      />
      <Sidebar
        graphId={graphId}
        graphState={graphState}
        selectedNode={selectedNode}
        onSelectNode={setSelectedNode}
        onRefresh={() => refreshGraph()}
      />
      <GraphCanvas
        graphState={graphState}
        analysis={analysis}
        selectedNode={selectedNode}
        onSelectNode={setSelectedNode}
      />
      <AnalysisPanel
        graphId={graphId}
        analysis={analysis}
        onRefresh={() => refreshGraph()}
      />
    </>
  );
}

// Mount
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
