from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status
from api.models import ServiceNode, DependencyEdge, SimulationRun, EvaluationReport

class SimulationAPITests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.valid_graph_payload = {
            "nodes": [
                {"id": "web", "name": "Web", "criticality": 1.0},
                {"id": "api", "name": "API", "criticality": 1.5},
                {"id": "db", "name": "Database", "criticality": 2.0}
            ],
            "edges": [
                {"source": "web", "target": "api", "amplification": 1.0},
                {"source": "api", "target": "db", "amplification": 2.0}
            ]
        }
        
    def test_initialize_graph(self):
        response = self.client.post('/api/initialize-graph', self.valid_graph_payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['nodes'], 3)
        self.assertEqual(response.data['edges'], 2)
        
        # Verify persistence
        self.assertEqual(ServiceNode.objects.count(), 3)
        self.assertEqual(DependencyEdge.objects.count(), 2)
        
    def test_malformed_graph_payload(self):
        bad_payload = {"nodes": [{"id": "web"}], "edges": []} # Missing required 'name' field
        response = self.client.post('/api/initialize-graph', bad_payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_full_simulation_flow(self):
        # 1. Init
        self.client.post('/api/initialize-graph', self.valid_graph_payload, format='json')
        
        # 2. Inject Fault inside DB
        fault_payload = {
            "target_node_id": "db",
            "fault_type": "latency_spike",
            "magnitude": 500.0,
            "duration_ticks": 3
        }
        resp = self.client.post('/api/inject-fault', fault_payload, format='json')
        self.assertEqual(resp.status_code, status.HTTP_202_ACCEPTED)
        
        # 3. Step 1 Tick (This initializes prediction)
        tick1_resp = self.client.post('/api/run-simulation')
        self.assertEqual(tick1_resp.status_code, status.HTTP_200_OK)
        
        # We can check risk via endpoint
        graph_state_resp = self.client.get('/api/graph-state')
        self.assertEqual(graph_state_resp.status_code, status.HTTP_200_OK)
        # Should contain some risk on 'db'
        self.assertGreater(graph_state_resp.data.get('db', 0.0), 0.0)
        
        # System risk endpoint
        sys_resp = self.client.get('/api/system-risk')
        self.assertEqual(sys_resp.status_code, status.HTTP_200_OK)
        # Assuming DB failing raises system severity above lowest level
        self.assertIn('severity_level', sys_resp.data)
        
        # 4. Step until stabilization (Wait 4 ticks to let fault expire and decay)
        for _ in range(4):
            self.client.post('/api/run-simulation')
            
        # 5. Get Evaluation Report
        eval_resp = self.client.get('/api/evaluation-report')
        self.assertEqual(eval_resp.status_code, status.HTTP_200_OK)
        self.assertIn('evaluation', eval_resp.data)
        self.assertIn('precision', eval_resp.data['evaluation'])
        
        # 6. Verify Postgres persistence of the report
        self.assertEqual(SimulationRun.objects.count(), 1)
        self.assertEqual(EvaluationReport.objects.count(), 1)
