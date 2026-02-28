from typing import List, Dict, Any
from simulation.graph_engine import GraphEngine
from simulation.models import ServiceNode as SimServiceNode, DependencyEdge as SimDependencyEdge
from simulation.metrics_generator import MetricsGenerator, FaultEvent
from simulation.propagation import RiskPropagationEngine
from simulation.risk_aggregation import CascadeSeverityScorer
from simulation.counterfactual_logger import CounterfactualLogger
from .models import ServiceNode, DependencyEdge, SimulationRun, EvaluationReport

class SimulationService:
    """
    Singleton orchestration layer for the pure Python simulation components.
    Ensures that Django views never touch NetworkX or internal math logic directly.
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
        self.risk_engine = RiskPropagationEngine()
        self.scorer = CascadeSeverityScorer()
        self.logger = CounterfactualLogger()
        self.current_tick = 0
        self.last_intelligence = {}
        self.last_risks = {}
    
    def reset_state(self):
        self._initialize_state()
        
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
                
        # 3. Propagate Risk
        self.last_risks = self.risk_engine.propagate_risk(self.graph_engine)
        
        # 4. Score Severity
        self.last_intelligence = self.scorer.aggregate(self.last_risks, self.graph_engine)
        
        # 5. Handle Logging
        if not self.logger.is_tracking:
            # We assume a fault was just injected if risk is high but we aren't tracking
            if self.last_intelligence.get("severity_level") != "LOW":
                self.logger.snapshot_prediction(self.current_tick, self.last_intelligence, self.last_risks)
        else:
            self.logger.track_actual_tick(self.current_tick, self.last_intelligence, self.last_risks)
            
            # If tracking just finished, persist the evaluation run to Postgres
            if not self.logger.is_tracking:
                self._persist_evaluation_run()
                
        return {
            "tick": self.current_tick,
            "intelligence": self.last_intelligence,
            "risks": self.last_risks
        }
        
    def _persist_evaluation_run(self):
        """
        Takes the counterfactual logger results and saves them to the DB.
        """
        eval_data = self.logger.evaluate()
        if "error" in eval_data:
            return
            
        metrics = eval_data.get("evaluation", {})
        
        # We don't have the explicit target node in the logger, but we can assign none for now or look at highest risk
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
        return self.last_risks
