import networkx as nx
from typing import List, Dict, Optional
from .models import ServiceNode, DependencyEdge

class GraphEngine:
    """
    Wrapper around NetworkX.DiGraph to manage the microservice topology.
    Nodes represent services, and directed edges represent dependencies (A depends on B).
    Risk propagates in reverse of dependency (If B fails, A is at risk).
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node: ServiceNode):
        self.graph.add_node(node.id, data=node)

    def add_edge(self, edge: DependencyEdge):
        # Directed edge: source_node depends on target_node
        # For risk propagation, if target fails, source is impacted.
        self.graph.add_edge(edge.source_node_id, edge.target_node_id, data=edge)

    def build_from_definitions(self, nodes: List[ServiceNode], edges: List[DependencyEdge]):
        self.graph.clear()
        for node in nodes:
            self.add_node(node)
        for edge in edges:
            self.add_edge(edge)

    def get_node(self, node_id: str) -> Optional[ServiceNode]:
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]['data']
        return None

    def get_edge(self, source_id: str, target_id: str) -> Optional[DependencyEdge]:
        if self.graph.has_edge(source_id, target_id):
            return self.graph.edges[source_id, target_id]['data']
        return None

    def get_upstream_dependents(self, node_id: str) -> List[str]:
        """
        Get all nodes that depend directly on the given node_id.
        In a graph where A -> B means A depends on B,
        if B fails, A is affected. We want to find A.
        Therefore, we look for predecessors of B.
        """
        if self.graph.has_node(node_id):
            return list(self.graph.predecessors(node_id))
        return []

    def get_downstream_dependencies(self, node_id: str) -> List[str]:
        """
        Get all nodes that the given node_id depends on.
        Look for successors of A.
        """
        if self.graph.has_node(node_id):
            return list(self.graph.successors(node_id))
        return []

    def update_node(self, node: ServiceNode):
        if self.graph.has_node(node.id):
            self.graph.nodes[node.id]['data'] = node

    def get_all_nodes(self) -> List[ServiceNode]:
        return [data['data'] for _, data in self.graph.nodes(data=True)]

    def get_all_edges(self) -> List[DependencyEdge]:
        return [data['data'] for _, _, data in self.graph.edges(data=True)]
