import random
import statistics
import copy
from typing import Dict, Any, List

from simulation.models import ServiceNode, DependencyEdge, MetricTick
from simulation.graph_engine import GraphEngine
from simulation.metrics_generator import MetricsGenerator, FaultEvent
from simulation.graph_learning_model import GraphFailurePredictor
from simulation.action_engine import PreventiveActionEngine
from simulation.counterfactual_logger import CounterfactualLogger

class BenchmarkRunner:
    """
    Executes structured experimental trials and produces quantitative resilience metrics
    suitable for research validation. Evaluates the difference between baseline,
    action mitigations, and adaptive graph-learning predictions.

    Uses GraphFailurePredictor (Graph Learning Agent) instead of the legacy
    deterministic RiskPropagationEngine + DeterministicFeedbackLoop.
    """
    def __init__(self, config: Dict[str, Any]):
        self.num_services = config.get("num_services", 10)
        self.density = config.get("density", 0.3)
        self.fault_type = config.get("fault_type", "cpu_spike")
        self.fault_duration = config.get("fault_duration", 3)
        self.learning_cycles = config.get("learning_cycles", 10)
        self.repeated_trials = config.get("repeated_trials", 3)
        
    def _generate_synthetic_graph(self, seed: int) -> GraphEngine:
        random.seed(seed)
        nodes = []
        for i in range(self.num_services):
            crit = random.choice([1.0, 1.5, 2.0])
            nodes.append(ServiceNode(id=f"node_{i}", service_name=f"Service_{i}", criticality_score=crit))
            
        edges = []
        for i in range(self.num_services):
            for j in range(self.num_services):
                if i != j and random.random() < self.density:
                    # Avoid back edges to keep it mostly DAG-like
                    if i < j:
                        edges.append(DependencyEdge(
                            source_node_id=f"node_{i}",
                            target_node_id=f"node_{j}",
                            amplification_factor=random.uniform(0.8, 2.5)
                        ))
        graph = GraphEngine()
        graph.build_from_definitions(nodes, edges)
        return graph

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Runs the benchmark comparing 3 modes:
        1. No mitigation (baseline)
        2. Static mitigation (action engine with cold-start predictor)
        3. Adaptive learning (predictor improves over cycles)
        """
        results_no_mitigation = []
        results_with_mitigation = []
        results_with_learning = []

        for trial in range(self.repeated_trials):
            # Shared predictor that learns across cycles
            learning_predictor = GraphFailurePredictor()
            
            for cycle in range(self.learning_cycles):
                seed = trial * 1000 + cycle
                
                # We need a stable target node per seed
                random.seed(seed)
                target_node = f"node_{random.randint(0, self.num_services - 1)}"
                
                # 1. No Mitigation & 2. Static Mitigation (fresh predictor each time)
                static_predictor = GraphFailurePredictor()
                static_res = self._run_single_scenario(seed, target_node, static_predictor, train_model=False)
                if static_res:
                    results_no_mitigation.append(static_res["baseline"])
                    results_with_mitigation.append(static_res["mitigated"])
                
                # 3. With Learning (shared predictor improves over cycles)
                adaptive_res = self._run_single_scenario(seed, target_node, learning_predictor, train_model=True)
                if adaptive_res:
                    results_with_learning.append(adaptive_res["mitigated"])

        return {
            "total_scenarios_per_track": self.repeated_trials * self.learning_cycles,
            "no_mitigation_metrics": self._aggregate(results_no_mitigation),
            "static_mitigation_metrics": self._aggregate(results_with_mitigation),
            "adaptive_learning_metrics": self._aggregate(results_with_learning, compute_trend=True, cycles=self.learning_cycles)
        }

    def _run_single_scenario(self, seed: int, target_node: str, predictor: GraphFailurePredictor, train_model: bool) -> Dict[str, Any]:
        graph = self._generate_synthetic_graph(seed)
        
        action_engine = PreventiveActionEngine(predictor)
        logger = CounterfactualLogger(high_risk_threshold=predictor.high_risk_threshold)
        metrics_gen = MetricsGenerator()
        
        metrics_gen.inject_fault(FaultEvent(
            node_id=target_node,
            fault_type=self.fault_type,
            magnitude=1.0,  
            duration_ticks=self.fault_duration
        ))
        
        tick = 0
        action_result = None
        last_risks = {}
        last_intel = {}
        
        while tick < 15:
            tick += 1
            nodes = graph.get_all_nodes()
            metrics = metrics_gen.generate_metrics(tick, nodes)
            for n in nodes:
                if n.id in metrics:
                    n.current_metrics = metrics[n.id]
                    graph.update_node(n)
                    
            # Use GraphFailurePredictor instead of old risk propagation
            prediction_result = predictor.predict_failure_probabilities(graph)
            risks = prediction_result["node_failure_probabilities"]
            intel = {
                "system_risk_score": prediction_result["system_risk_score"],
                "high_risk_node_count": prediction_result["high_risk_node_count"],
                "predicted_cascade_size": prediction_result["predicted_cascade_size"],
                "severity_level": prediction_result["severity_level"],
                "high_risk_nodes": prediction_result["high_risk_nodes"],
            }
            last_risks = risks
            last_intel = intel
            
            if not logger.is_tracking:
                if intel.get("severity_level") != "LOW":
                    logger.snapshot_prediction(tick, intel, risks)
                    action_result = action_engine.trigger_proactive_defense(graph, risks, intel)
            else:
                logger.track_actual_tick(tick, intel, risks)
                if not logger.is_tracking:
                    break
                    
        eval_report = logger.evaluate()
        if "error" in eval_report:
            return None
        
        # Train the predictor if in learning mode
        if train_model:
            actual_failed = logger.actual_failed_nodes
            actual_labels = {}
            for node in graph.get_all_nodes():
                actual_labels[node.id] = node.id in actual_failed
            predictor.train_on_batch(last_risks, actual_labels, graph)
            predictor.update_model()
            
        # The true "baseline" outcome is what the logger actually observed.
        base_cascade = eval_report["actual"]["actual_cascade_size"]
        base_severity = eval_report["evaluation"]["actual_inferred_severity"]
        
        mitigated_cascade = action_result["post_action_cascade_size"] if action_result and action_result.get("action_taken") != "NONE" else base_cascade
        mitigated_severity = action_result["severity_after"] if action_result and action_result.get("action_taken") != "NONE" else base_severity
        reduction = action_result["risk_reduction_percent"] if action_result and action_result.get("action_taken") != "NONE" else 0.0
        
        return {
            "baseline": {
                "cascade_size": base_cascade,
                "severity": base_severity,
                "precision": eval_report["evaluation"]["precision"],
                "recall": eval_report["evaluation"]["recall"],
                "stabilization_ticks": eval_report["actual"]["time_to_stabilization_ticks"],
                "cycle": seed % max(1, self.learning_cycles)
            },
            "mitigated": {
                "cascade_size": mitigated_cascade,
                "severity": mitigated_severity,
                "reduction_percent": reduction,
                "precision": eval_report["evaluation"]["precision"],
                "recall": eval_report["evaluation"]["recall"],
                "stabilization_ticks": eval_report["actual"]["time_to_stabilization_ticks"],
                "cycle": seed % max(1, self.learning_cycles)
            }
        }

    def _aggregate(self, results_list: List[Dict[str, Any]], compute_trend=False, cycles=1) -> Dict[str, Any]:
        if not results_list:
            return {}
            
        avg_cascade = statistics.mean([r["cascade_size"] for r in results_list])
        avg_precision = statistics.mean([r["precision"] for r in results_list])
        avg_recall = statistics.mean([r["recall"] for r in results_list])
        avg_stab = statistics.mean([r["stabilization_ticks"] for r in results_list])
        
        reduction = statistics.mean([r.get("reduction_percent", 0.0) for r in results_list])
        
        sev = [r["severity"] for r in results_list]
        sev_dist = {
            "CRITICAL": sev.count("CRITICAL"),
            "HIGH": sev.count("HIGH"),
            "MODERATE": sev.count("MODERATE"),
            "LOW": sev.count("LOW")
        }
        
        trend = []
        if compute_trend:
            for c in range(cycles):
                c_precs = [r["precision"] for r in results_list if r.get("cycle") == c]
                if c_precs:
                    trend.append(round(statistics.mean(c_precs), 3))
                    
        return {
            "average_cascade_size": round(avg_cascade, 2),
            "average_cascade_reduction_percent": round(reduction, 2),
            "average_precision": round(avg_precision, 4),
            "average_recall": round(avg_recall, 4),
            "average_stabilization_cycles": round(avg_stab, 2),
            "convergence_trend": trend,
            "severity_distribution": sev_dist
        }
