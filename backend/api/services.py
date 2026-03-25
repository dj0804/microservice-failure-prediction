from typing import List, Dict, Any
from simulation.graph_engine import GraphEngine
from simulation.models import ServiceNode as SimServiceNode, DependencyEdge as SimDependencyEdge
from simulation.metrics_generator import MetricsGenerator, FaultEvent
from simulation.graph_learning_model import GraphFailurePredictor
from simulation.counterfactual_logger import CounterfactualLogger
from .models import ServiceNode, DependencyEdge, SimulationRun, EvaluationReport

class SimulationService:
    """
    Singleton orchestration layer for the pure Python simulation components.
    Ensures that Django views never touch NetworkX or internal math logic directly.

    Uses the GraphFailurePredictor (Graph Learning Agent) for probabilistic
    failure prediction, replacing the legacy deterministic RiskPropagationEngine
    and CascadeSeverityScorer.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_state()
        return cls._instance
        
    def _initialize_state(self):
        self.graph_engine = GraphEngine()
        self.metrics_generator = MetricsGenerator()
        self.predictor = GraphFailurePredictor()
        self.logger = CounterfactualLogger()
        self.current_tick = 0
        self.last_intelligence = {}
        self.last_risks = {}
    
    def reset_state(self):
        self._initialize_state()

    def initialize_synthetic_graph(self, num_services: int, density: float):
        import networkx as nx
        import random
        G = nx.erdos_renyi_graph(num_services, density, directed=True, seed=random.randint(1, 10000))
        
        sim_nodes = []
        for i in range(num_services):
            sim_nodes.append(SimServiceNode(
                id=f"node_{i}",
                service_name=f"Service-{i}",
                criticality_score=random.choice([1.0, 1.2, 1.5, 2.0])
            ))
            
        sim_edges = []
        for source, target in G.edges():
            sim_edges.append(SimDependencyEdge(
                source_node_id=f"node_{source}",
                target_node_id=f"node_{target}",
                amplification_factor=round(random.uniform(0.5, 1.5), 2)
            ))
            
        self.graph_engine.build_from_definitions(sim_nodes, sim_edges)
        self.current_tick = 0
        self.last_risks = {}
        self.last_intelligence = {}
        self.logger = CounterfactualLogger()
        return {"status": "Graph initialized synthetically", "nodes": len(sim_nodes), "edges": len(sim_edges)}
        
    def initialize_graph(self, nodes_data: List[Dict], edges_data: List[Dict]):
        """
        Loads the topology into the generic GraphEngine and optionally persists it 
        to PostgreSQL for historical tracking.
        """
        sim_nodes = []
        for n in nodes_data:
            sim_nodes.append(SimServiceNode(
                id=n['id'], 
                service_name=n['name'], 
                criticality_score=n.get('criticality', 1.0)
            ))
            # Persist to DB
            ServiceNode.objects.update_or_create(
                node_id=n['id'],
                defaults={'name': n['name'], 'criticality': n.get('criticality', 1.0)}
            )
            
        sim_edges = []
        for e in edges_data:
            sim_edges.append(SimDependencyEdge(
                source_node_id=e['source'], 
                target_node_id=e['target'], 
                amplification_factor=e.get('amplification', 1.0)
            ))
            # Persist to DB
            source_obj = ServiceNode.objects.get(node_id=e['source'])
            target_obj = ServiceNode.objects.get(node_id=e['target'])
            DependencyEdge.objects.update_or_create(
                source=source_obj, 
                target=target_obj,
                defaults={'amplification': e.get('amplification', 1.0)}
            )
            
        self.graph_engine.build_from_definitions(sim_nodes, sim_edges)
        
        # Reset tick when a new graph is loaded
        self.current_tick = 0
        return {"status": "Graph initialized", "nodes": len(sim_nodes), "edges": len(sim_edges)}

    def inject_fault(self, node_id: str, fault_type: str, magnitude: float, duration_ticks: int):
        self.metrics_generator.inject_fault(FaultEvent(
            node_id=node_id,
            fault_type=fault_type,
            magnitude=magnitude,
            duration_ticks=duration_ticks
        ))
        return {"status": "Fault injected", "target": node_id}

    def run_simulation(self):
        """
        Advances the simulation clock safely orchestrating all components.

        Uses GraphFailurePredictor (Graph Learning Agent) for probabilistic
        failure prediction instead of the legacy deterministic propagation.
        """
        self.current_tick += 1
        nodes = self.graph_engine.get_all_nodes()
        
        # 1. Generate metrics for this tick
        metrics = self.metrics_generator.generate_metrics(self.current_tick, nodes)
        
        # 2. Update nodes in graph
        for node in nodes:
            if node.id in metrics:
                node.current_metrics = metrics[node.id]
                self.graph_engine.update_node(node)
                
        # 3. Predict failure probabilities (replaces propagate_risk + aggregate)
        prediction_result = self.predictor.predict_failure_probabilities(self.graph_engine)
        cascade_result = self.predictor.predict_cascade(
            self.graph_engine,
            prediction_result["node_failure_probabilities"],
        )
        
        # Extract backward-compatible fields
        self.last_risks = prediction_result["node_failure_probabilities"]
        self.last_intelligence = {
            "system_risk_score": prediction_result["system_risk_score"],
            "high_risk_node_count": prediction_result["high_risk_node_count"],
            "predicted_cascade_size": prediction_result["predicted_cascade_size"],
            "severity_level": prediction_result["severity_level"],
            "prediction_mode": prediction_result["prediction_mode"],
            "high_risk_nodes": prediction_result["high_risk_nodes"],
            "predicted_affected_nodes": cascade_result["predicted_affected_nodes"],
            "cascade_size": cascade_result["cascade_size"],
            "propagation_paths": cascade_result["propagation_paths"],
            "propagation_risk_score": cascade_result["propagation_risk_score"],
            "system_failure_probability": cascade_result["system_failure_probability"],
        }
        
        # 4. Handle Logging & Training Integration
        if not self.logger.is_tracking:
            # Start tracking if severity is non-trivial
            if self.last_intelligence.get("severity_level") != "LOW":
                self.logger.snapshot_prediction(self.current_tick, self.last_intelligence, self.last_risks)
        else:
            self.logger.track_actual_tick(self.current_tick, self.last_intelligence, self.last_risks)
            
            # If tracking just finished, train the model and persist evaluation
            if not self.logger.is_tracking:
                self._train_and_persist()
                
        return {
            "tick": self.current_tick,
            "intelligence": self.last_intelligence,
            "risks": self.last_risks
        }

    def _train_and_persist(self):
        """
        After the CounterfactualLogger finishes tracking an event:
        1. Extract predicted vs actual failure labels
        2. Feed to predictor.train_on_batch()
        3. Retrain model via predictor.update_model()
        4. Persist evaluation to Postgres
        """
        eval_data = self.logger.evaluate()
        if "error" in eval_data:
            return
        
        # Build actual failure labels from logger
        actual_failed_nodes = self.logger.actual_failed_nodes
        predicted_probs = {}
        actual_labels = {}
        
        for node in self.graph_engine.get_all_nodes():
            predicted_probs[node.id] = self.last_risks.get(node.id, 0.0)
            actual_labels[node.id] = node.id in actual_failed_nodes
        
        # Train the Graph Learning Agent
        batch_result = self.predictor.train_on_batch(
            predicted_probs, actual_labels, self.graph_engine
        )
        if batch_result["ready_to_train"]:
            self.predictor.update_model()
        
        # Persist evaluation to DB
        self._persist_evaluation_run(eval_data)

    def _persist_evaluation_run(self, eval_data: Dict = None):
        """
        Takes the counterfactual logger results and saves them to the DB.
        """
        if eval_data is None:
            eval_data = self.logger.evaluate()
        if "error" in eval_data:
            return
            
        metrics = eval_data.get("evaluation", {})
        
        run_obj = SimulationRun.objects.create(
            fault_type="synthetic_cascade", 
            final_severity=metrics.get("actual_inferred_severity", "LOW"),
            cascade_size=eval_data.get("actual", {}).get("actual_cascade_size", 0),
            max_system_risk=eval_data.get("actual", {}).get("max_system_risk_observed", 0.0)
        )
        
        EvaluationReport.objects.create(
            simulation_run=run_obj,
            cascade_size_error=metrics.get("cascade_size_error", 0),
            severity_match=metrics.get("severity_match", True),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            binary_accuracy=metrics.get("binary_accuracy", True)
        )

    def get_system_risk(self):
        return self.last_intelligence
        
    def get_evaluation_report(self):
        return self.logger.evaluate()
        
    def get_graph_state(self):
        nodes = []
        for n in self.graph_engine.get_all_nodes():
            nodes.append({
                "id": n.id,
                "name": n.service_name,
                "calculated_risk_score": self.last_risks.get(n.id, 0.0),
                "criticality_score": n.criticality_score
            })
            
        edges = []
        for e in self.graph_engine.get_all_edges():
            edges.append({
                "source": e.source_node_id,
                "target": e.target_node_id,
                "amplification_factor": e.amplification_factor
            })
            
        return {"nodes": nodes, "edges": edges}
