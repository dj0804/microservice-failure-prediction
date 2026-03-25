from typing import Dict, Any, List
import copy
from .graph_engine import GraphEngine
from .graph_learning_model import GraphFailurePredictor

class PreventiveActionEngine:
    """
    Evaluates predicted cascades and applies hypothetical graph modifications
    (ISOLATE, THROTTLE) to a temporary topological copy, measuring the hypothetical
    risk reduction before returning an optimal mitigation strategy.

    Uses GraphFailurePredictor (Graph Learning Agent) for probabilistic evaluation
    instead of the legacy deterministic RiskPropagationEngine + CascadeSeverityScorer.
    """
    
    def __init__(self, predictor: GraphFailurePredictor):
        self.predictor = predictor

    def _clone_graph(self, graph: GraphEngine) -> GraphEngine:
        """
        Creates a deep but efficient copy of the graph to run synthetic mitigations
        without permanently modifying the real topology.
        """
        cloned = GraphEngine()
        cloned.graph = graph.graph.copy()
        
        # Deep copy the node objects inside the cloned graph as well because 
        # the propagation relies on node object states
        for nid in cloned.graph.nodes:
            cloned.graph.nodes[nid]['data'] = copy.deepcopy(cloned.graph.nodes[nid]['data'])
            
        for u, v in cloned.graph.edges:
            cloned.graph.edges[u, v]['data'] = copy.deepcopy(cloned.graph.edges[u, v]['data'])
            
        return cloned

    def apply_action(self, graph: GraphEngine, action_type: str, target_nodes: List[str]) -> GraphEngine:
        """
        Applies a specific action to a cloned graph.
        - ISOLATE: Removes all outgoing edges from the target node. (Dependents can no longer reach it)
        - THROTTLE: Halves the amplification_factor on outgoing edges.
        """
        temp_graph = self._clone_graph(graph)
        
        for node_id in target_nodes:
            # We want to protect dependents OF the failing node.
            # In our model: source_node depends_on target_node (e.g., Web -> API).
            # So if target_node (API) is failing, the outgoing edges FROM target_node 
            # don't exist in that direction. We must find edges *pointing to* target_node.
            # NetworkX `in_edges(node_id)` gives all (u, v) where v == node_id.
            
            incoming_edges = list(temp_graph.graph.in_edges(node_id))
            
            for src, dst in incoming_edges:
                if action_type == "ISOLATE":
                    # Simulate circuit breaker open
                    temp_graph.graph.remove_edge(src, dst)
                elif action_type == "THROTTLE":
                    # Simulate rate limiting/fallback
                    edge_data = temp_graph.get_edge(src, dst)
                    if edge_data:
                        edge_data.amplification_factor *= 0.5
                            
        return temp_graph

    def evaluate_mitigation(self, 
                            original_graph: GraphEngine, 
                            baseline_intelligence: Dict[str, Any],
                            action_type: str, 
                            target_nodes: List[str]) -> Dict[str, Any]:
        """
        Runs the full mitigation evaluation cycle using the Graph Learning Agent.
        """
        baseline_cascade_size = baseline_intelligence.get("predicted_cascade_size", 0)
        baseline_severity = baseline_intelligence.get("severity_level", "LOW")
        baseline_sys_risk = baseline_intelligence.get("system_risk_score", 0.0)

        # 1. Apply action to temporary graph
        temp_graph = self.apply_action(original_graph, action_type, target_nodes)
        
        # 2. Predict failure probabilities on modified graph
        new_result = self.predictor.predict_failure_probabilities(temp_graph)
        
        new_cascade_size = new_result.get("predicted_cascade_size", 0)
        new_severity = new_result.get("severity_level", "LOW")
        new_sys_risk = new_result.get("system_risk_score", 0.0)

        # 3. Measure risk reduction
        risk_reduction_percent = 0.0
        if baseline_sys_risk > 0:
            risk_reduction_percent = ((baseline_sys_risk - new_sys_risk) / baseline_sys_risk) * 100.0
            
        risk_reduction_percent = max(0.0, risk_reduction_percent)

        return {
            "baseline_cascade_size": baseline_cascade_size,
            "post_action_cascade_size": new_cascade_size,
            "risk_reduction_percent": round(risk_reduction_percent, 2),
            "action_taken": action_type,
            "severity_before": baseline_severity,
            "severity_after": new_severity
        }

    def trigger_proactive_defense(self, graph: GraphEngine, risks: Dict[str, float], intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Top level orchestrator. Decides if action is needed, simulates multiple, picks the best.
        """
        severity = intelligence.get("severity_level", "LOW")
        if severity not in ["HIGH", "CRITICAL"]:
            return {
                "action_taken": "NONE",
                "reason": "Severity below threshold"
            }
            
        # Get high-risk nodes from the intelligence output
        high_risk_nodes = intelligence.get("high_risk_nodes", [])
        
        # Fallback: derive from raw probabilities if not in intelligence
        if not high_risk_nodes:
            high_risk_nodes = [nid for nid, prob in risks.items() if prob >= self.predictor.high_risk_threshold]
        
        if not high_risk_nodes:
             return {"action_taken": "NONE", "reason": "No discrete high risk nodes identified"}

        # Simulate both actions
        throttle_eval = self.evaluate_mitigation(graph, intelligence, "THROTTLE", high_risk_nodes)
        isolate_eval = self.evaluate_mitigation(graph, intelligence, "ISOLATE", high_risk_nodes)
        
        # Simple heuristic: Pick what reduces cascade the most, preferring Throttle if equal (less destructive)
        if isolate_eval["post_action_cascade_size"] < throttle_eval["post_action_cascade_size"]:
            return isolate_eval
            
        if throttle_eval["risk_reduction_percent"] > 0:
            return throttle_eval
            
        return isolate_eval
