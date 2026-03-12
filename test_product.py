"""TCA Calculator — End-to-end integration tests.

Uses FastAPI TestClient (no running server needed).
"""

import unittest

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestGraphCRUD(unittest.TestCase):

    def test_create_graph(self):
        r = client.post('/api/graphs', json={'name': 'Test Graph'})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn('graph_id', data)
        self.assertEqual(data['name'], 'Test Graph')

    def test_add_nodes_and_edges(self):
        gid = client.post('/api/graphs', json={'name': 'crud'}).json()['graph_id']

        # Add nodes.
        r = client.post(f'/api/graphs/{gid}/nodes', json={'label': 'Node A', 'node_id': 'a'})
        self.assertEqual(r.json()['node_id'], 'a')
        client.post(f'/api/graphs/{gid}/nodes', json={'label': 'Node B', 'node_id': 'b'})

        # Add edge.
        r = client.post(f'/api/graphs/{gid}/edges', json={
            'source_id': 'a', 'target_id': 'b', 'edge_type': 'SEEKS',
        })
        self.assertEqual(r.status_code, 200)

        # Verify state.
        state = client.get(f'/api/graphs/{gid}').json()
        self.assertEqual(state['node_count'], 2)
        self.assertEqual(state['edge_count'], 1)

    def test_delete_node_cleans_edges(self):
        gid = client.post('/api/graphs', json={'name': 'del'}).json()['graph_id']
        client.post(f'/api/graphs/{gid}/nodes', json={'label': 'X', 'node_id': 'x'})
        client.post(f'/api/graphs/{gid}/nodes', json={'label': 'Y', 'node_id': 'y'})
        client.post(f'/api/graphs/{gid}/nodes', json={'label': 'Z', 'node_id': 'z'})
        client.post(f'/api/graphs/{gid}/edges', json={
            'source_id': 'x', 'target_id': 'y', 'edge_type': 'MIRRORS',
        })
        client.post(f'/api/graphs/{gid}/edges', json={
            'source_id': 'z', 'target_id': 'y', 'edge_type': 'VERIFIES',
        })

        # Delete Y — should clean up edges from X and Z to Y.
        r = client.delete(f'/api/graphs/{gid}/nodes/y')
        self.assertEqual(r.status_code, 200)

        state = client.get(f'/api/graphs/{gid}').json()
        self.assertEqual(state['node_count'], 2)
        self.assertEqual(state['edge_count'], 0)  # Both edges removed.

    def test_list_and_delete_graph(self):
        gid = client.post('/api/graphs', json={'name': 'to_delete'}).json()['graph_id']
        graphs = client.get('/api/graphs').json()
        ids = [g['graph_id'] for g in graphs]
        self.assertIn(gid, ids)

        client.delete(f'/api/graphs/{gid}')
        graphs = client.get('/api/graphs').json()
        ids = [g['graph_id'] for g in graphs]
        self.assertNotIn(gid, ids)


class TestAnalysis(unittest.TestCase):

    def test_analyze_finds_problems(self):
        gid = client.post('/api/graphs', json={'name': 'analysis'}).json()['graph_id']
        client.post(f'/api/graphs/{gid}/nodes', json={'label': 'GDP', 'node_id': 'gdp'})
        client.post(f'/api/graphs/{gid}/nodes', json={'label': 'Wellbeing', 'node_id': 'well'})
        client.post(f'/api/graphs/{gid}/edges', json={
            'source_id': 'gdp', 'target_id': 'well', 'edge_type': 'SEEKS',
        })

        r = client.post(f'/api/graphs/{gid}/analyze')
        data = r.json()
        self.assertIn('health', data)
        self.assertIn('confidence', data)
        self.assertIn('solutions', data)
        self.assertGreater(data['solution_count'], 0)

        # Should find dead end and ungrounded SEEKS.
        types = {s['problem_type'] for s in data['solutions']}
        self.assertIn('dead_end', types)
        self.assertIn('ungrounded', types)

    def test_apply_solution_modifies_graph(self):
        gid = client.post('/api/graphs', json={'name': 'apply'}).json()['graph_id']
        client.post(f'/api/graphs/{gid}/nodes', json={'label': 'A', 'node_id': 'a'})
        client.post(f'/api/graphs/{gid}/nodes', json={'label': 'B', 'node_id': 'b'})
        client.post(f'/api/graphs/{gid}/edges', json={
            'source_id': 'a', 'target_id': 'b', 'edge_type': 'SEEKS',
        })

        # Analyze first (populates cached solutions).
        client.post(f'/api/graphs/{gid}/analyze')

        # Get initial edge count.
        state_before = client.get(f'/api/graphs/{gid}').json()
        edges_before = state_before['edge_count']

        # Apply first solution.
        r = client.post(f'/api/graphs/{gid}/solutions/0/apply')
        self.assertEqual(r.status_code, 200)

        # Graph should have changed.
        state_after = client.get(f'/api/graphs/{gid}').json()
        self.assertNotEqual(state_after['edge_count'], edges_before)

    def test_empty_graph_analysis(self):
        gid = client.post('/api/graphs', json={'name': 'empty'}).json()['graph_id']
        r = client.post(f'/api/graphs/{gid}/analyze')
        data = r.json()
        self.assertEqual(data['node_count'], 0)
        self.assertEqual(data['solution_count'], 0)


class TestTemplates(unittest.TestCase):

    def test_list_templates(self):
        r = client.get('/api/templates')
        data = r.json()
        names = [t['name'] for t in data['templates']]
        self.assertIn('economics', names)
        self.assertIn('blank', names)

    def test_load_economics_template(self):
        r = client.post('/api/graphs', json={'name': 'econ', 'template': 'economics'})
        gid = r.json()['graph_id']
        state = client.get(f'/api/graphs/{gid}').json()
        self.assertGreater(state['node_count'], 10)
        self.assertGreater(state['edge_count'], 20)

    def test_analyze_template(self):
        gid = client.post('/api/graphs', json={'name': 'econ', 'template': 'economics'}).json()['graph_id']
        r = client.post(f'/api/graphs/{gid}/analyze')
        data = r.json()
        self.assertGreater(data['solution_count'], 5)

    def test_all_templates_load(self):
        templates = client.get('/api/templates').json()['templates']
        for t in templates:
            r = client.post('/api/graphs', json={'name': t['name'], 'template': t['name']})
            gid = r.json()['graph_id']
            state = client.get(f'/api/graphs/{gid}').json()
            self.assertEqual(state['node_count'], t['node_count'])


class TestExport(unittest.TestCase):

    def test_export_state(self):
        gid = client.post('/api/graphs', json={'name': 'exp', 'template': 'economics'}).json()['graph_id']
        r = client.get(f'/api/graphs/{gid}/export/state')
        data = r.json()
        self.assertEqual(data['protocol_version'], '0.1.0')
        self.assertIn('L2_graph', data)

    def test_export_boot_has_no_private_data(self):
        gid = client.post('/api/graphs', json={'name': 'boot', 'template': 'economics'}).json()['graph_id']
        r = client.get(f'/api/graphs/{gid}/export/boot')
        boot = r.json()
        self.assertEqual(boot['protocol_version'], '0.1.0')
        self.assertIn('architecture', boot)
        self.assertEqual(len(boot['architecture']['gates']), 7)

        # Boot protocol must NOT contain node labels/data.
        boot_str = str(boot)
        self.assertNotIn('Federal Reserve', boot_str)
        self.assertNotIn('Wellbeing', boot_str)
        self.assertNotIn('GDP', boot_str)


class TestEdgeValidation(unittest.TestCase):

    def test_invalid_edge_type_rejected(self):
        gid = client.post('/api/graphs', json={'name': 'val'}).json()['graph_id']
        client.post(f'/api/graphs/{gid}/nodes', json={'label': 'A', 'node_id': 'a'})
        client.post(f'/api/graphs/{gid}/nodes', json={'label': 'B', 'node_id': 'b'})
        r = client.post(f'/api/graphs/{gid}/edges', json={
            'source_id': 'a', 'target_id': 'b', 'edge_type': 'INVALID_TYPE',
        })
        self.assertEqual(r.status_code, 422)

    def test_nonexistent_graph_returns_404(self):
        r = client.get('/api/graphs/nonexistent')
        self.assertEqual(r.status_code, 404)


if __name__ == '__main__':
    unittest.main()
