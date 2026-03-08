from typing import Dict, List, Any
from .graph_engine import GraphEngine

class CascadeSeverityScorer:
    """
    Analyzes propagated risk scores across the topology to generate 
    system-level predictive intelligence and severity classification.
    """
    def __init__(self, high_risk_threshold: float = 0.5):
        self.high_risk_threshold = high_risk_threshold
        
    def calculate_system_risk_score(self, risks: Dict[str, float], graph: GraphEngine) -> float:
        """
        Computes the weighted mean of normalized risk scores using service criticality.
        """
        nodes = graph.get_all_nodes()
        if not nodes:
            return 0.0
            
        total_risk_weight = 0.0
        total_criticality = 0.0
        
        for node in nodes:
            # Look up risk, defaulting to 0.0
            risk = risks.get(node.id, 0.0)
            total_risk_weight += risk * node.criticality_score
            total_criticality += node.criticality_score
            
        if total_criticality == 0.0:
            return 0.0
            
        return total_risk_weight / total_criticality
        
    def calculate_risk_concentration(self, risks: Dict[str, float]) -> float:
        """
        Calculates how localized the risk is. 
        Returns max_risk / sum_risk. 
        Close to 1.0 = Highly localized (one node holds most risk).
        Close to 0.0 = Widely distributed (risk spread across many nodes).
        """
        vals = list(risks.values())
        sum_risk = sum(vals)
        if sum_risk == 0:
            return 0.0
        return max(vals) / sum_risk

    def get_reachable_cascade_nodes(self, start_nodes: List[str], graph: GraphEngine) -> set:
        """
        Finds all nodes that can be impacted downstream in the failure cascade.
        Since risk flows upstream (dependents are impacted by failures in their dependencies),
        we traverse upstream dependents in the graph.
        """
        visited = set()
        queue = list(start_nodes)
        
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                # If an upstream node depends on 'curr', it is at risk when 'curr' fails
                dependents = graph.get_upstream_dependents(curr)
                queue.extend(dependents)
                
        return visited

    def find_critical_paths(self, start_nodes: List[str], graph: GraphEngine) -> List[List[str]]:
        """
        Finds the longest dependency paths starting from high-risk nodes, 
        illustrating the critical path of the cascade.
        """
        if not start_nodes:
            return []
            
        longest_paths = []
        max_len = 0
        
        for start_node in start_nodes:
            # Depth-First Search to explore all dependency chains
            stack = [(start_node, [start_node])]
            
            while stack:
                curr, path = stack.pop()
                dependents = graph.get_upstream_dependents(curr)
                
                # Filter dependencies to avoid graph cycles
                valid_deps = [d for d in dependents if d not in path]
                
                if not valid_deps:
                    # Leaf of this traversal path
                    if len(path) > max_len:
                        max_len = len(path)
                        longest_paths = [path]
                    elif len(path) == max_len and path not in longest_paths:
                        longest_paths.append(path)
                else:
                    for d in valid_deps:
                        stack.append((d, path + [d]))
                        
        return longest_paths

    def classify_severity(self, system_score: float, high_risk_count: int, cascade_size: int, concentration: float) -> str:
        """
        Rule-based Severity Classification.
        - CRITICAL: High systemic threat, multiple nodes at high risk, or large propagating cascade.
        - HIGH: Moderate systemic threat, propagating to at least a few nodes.
        - MODERATE: Localized issue or low-level distributed risk.
        - LOW: Baseline operational noise.
        """
        if system_score >= 0.7 or cascade_size >= 4 or high_risk_count >= 3:
            return "CRITICAL"
        if system_score >= 0.4 or cascade_size >= 2 or high_risk_count >= 2:
            return "HIGH"
        if system_score >= 0.1 or high_risk_count >= 1:
            return "MODERATE"
        return "LOW"

    def aggregate(self, risks: Dict[str, float], graph: GraphEngine) -> Dict[str, Any]:
        """
        Takes raw node risks and outputs system-level predictive intelligence.
        """
        system_risk_score = self.calculate_system_risk_score(risks, graph)
        concentration_score = self.calculate_risk_concentration(risks)
        
        high_risk_nodes = [nid for nid, r in risks.items() if r >= self.high_risk_threshold]
        high_risk_node_count = len(high_risk_nodes)
        
        cascade_set = self.get_reachable_cascade_nodes(high_risk_nodes, graph)
        predicted_cascade_size = len(cascade_set)
        
        critical_paths = self.find_critical_paths(high_risk_nodes, graph)
        
        severity_level = self.classify_severity(
            system_risk_score, 
            high_risk_node_count, 
            predicted_cascade_size,
            concentration_score
        )
        
        return {
            "system_risk_score": round(system_risk_score, 4),
            "high_risk_node_count": high_risk_node_count,
            "predicted_cascade_size": predicted_cascade_size,
            "critical_paths": critical_paths,
            "risk_concentration_score": round(concentration_score, 4),
            "severity_level": severity_level
        }
