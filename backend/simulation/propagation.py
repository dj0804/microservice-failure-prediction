"""
===============================================================================
DEPRECATED — This module is superseded by graph_learning_model.py

The RiskPropagationEngine (BFS + hop decay) has been replaced by the
GraphFailurePredictor (Graph Learning Agent), which learns failure
propagation implicitly via graph-aware embeddings rather than explicit
BFS traversal.

This module is retained for backward compatibility and testing of the
legacy deterministic system.
===============================================================================
"""
import warnings
warnings.warn(
    "simulation.propagation is deprecated. Use simulation.graph_learning_model.GraphFailurePredictor instead.",
    DeprecationWarning,
    stacklevel=2,
)

from typing import Dict
from .graph_engine import GraphEngine
from .models import ServiceNode

class RiskPropagationEngine:
    """
    Deterministic engine that diffuses localized risk across the NetworkX topology.
    This adheres to the TRL-4 simulated model of cascade propagation.
    """
    def __init__(self, hop_decay_factor: float = 0.5, amplification_multiplier: float = 1.0):
        self.hop_decay_factor = hop_decay_factor
        self.amplification_multiplier = amplification_multiplier

    def calculate_base_risk(self, node: ServiceNode) -> float:
        """
        Calculate localized risk based on a node's current metrics.
        Returns a value typically between 0 and 1.
        """
        if not node.current_metrics:
            return 0.0

        m = node.current_metrics
        # Heuristic math for validation:
        # High CPU or High Error Rate or High Latency = high risk
        # Normalize arbitrarily for prototype. 
        # Ex: cpu > 0.8 is bad. latency > 200ms is bad, error > 0.05 is bad.
        risk = 0.0
        
        if m.cpu_utilization > 0.8:
            risk += (m.cpu_utilization - 0.8) * 2  # Max ~0.4
        
        if m.latency_ms > 200:
            risk += min(0.3, (m.latency_ms - 200) / 1000) # Max 0.3
            
        if m.error_rate > 0.05:
            risk += min(0.4, (m.error_rate - 0.05) * 4) # Max 0.4
            
        # Bound base risk between 0 and 1
        base_risk = min(1.0, max(0.0, risk))
        
        # Apply criticality
        return base_risk * node.criticality_score

    def propagate_risk(self, graph_engine: GraphEngine) -> Dict[str, float]:
        """
        Calculates localized risk for all nodes, then diffuses it UP the dependency chain.
        (If database fails, auth-service is at risk. Data flows DB -> Auth, depend is Auth -> DB).
        """
        nodes = graph_engine.get_all_nodes()
        
        # 1. Compute Base Risk
        local_risks = {n.id: self.calculate_base_risk(n) for n in nodes}
        propagated_risks = {n.id: 0.0 for n in nodes}
        
        # 2. Propagate Risk (Graph Traversal)
        # For a true DAG, we would do topological sort. 
        # For simplicity and cyclical graphs, we can do a bounded breadth-first traversal 
        # from each node that has a non-zero local risk.
        for source_node_id, initial_risk in local_risks.items():
            if initial_risk <= 0.01:
                continue
                
            # Diffuse from source up to dependents.
            # E.g. source_node_id is failing. Which nodes depend on it?
            queue = [(source_node_id, initial_risk, 0)]  # (node, current_risk_magnitude, distance)
            visited = set()
            
            while queue:
                current_id, risk_magnitude, depth = queue.pop(0)
                
                # Add risk to the node
                propagated_risks[current_id] += risk_magnitude
                visited.add(current_id)
                
                # Bounded depth to prevent infinite cyclical loops (max 5 hops)
                if depth >= 5:
                    continue
                    
                # Find nodes that depend on current_id
                dependents = graph_engine.get_upstream_dependents(current_id)
                for dep_id in dependents:
                    if dep_id not in visited:
                        # Edge from dep_id -> current_id
                        edge = graph_engine.get_edge(dep_id, current_id)
                        amp_factor = edge.amplification_factor if edge else 1.0
                        
                        # Calculate diffused risk to dependent node
                        next_risk = risk_magnitude * amp_factor * self.amplification_multiplier * self.hop_decay_factor
                        
                        if next_risk > 0.01:
                            queue.append((dep_id, next_risk, depth + 1))
                            
        # Update graph nodes with final risk
        for node in nodes:
            node.calculated_risk_score = propagated_risks[node.id]
            graph_engine.update_node(node)
            
        return propagated_risks
