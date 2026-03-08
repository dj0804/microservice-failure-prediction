import pytest
from src.simulation.models import ServiceNode, DependencyEdge
from src.simulation.graph_engine import GraphEngine
from src.simulation.risk_aggregation import CascadeSeverityScorer

@pytest.fixture
def base_graph():
    """
    Topology: 
    Web1 \\
          --> API --> DB
    Web2 /
    """
    engine = GraphEngine()
    nodes = [
        ServiceNode(id="web1", service_name="Web 1", criticality_score=1.0),
        ServiceNode(id="web2", service_name="Web 2", criticality_score=1.0),
        ServiceNode(id="api", service_name="API Gateway", criticality_score=1.5),
        ServiceNode(id="db", service_name="Database", criticality_score=2.0)
    ]
    edges = [
        DependencyEdge(source_node_id="web1", target_node_id="api"),
        DependencyEdge(source_node_id="web2", target_node_id="api"),
        DependencyEdge(source_node_id="api", target_node_id="db")
    ]
    engine.build_from_definitions(nodes, edges)
    return engine

def test_localized_leaf_failure_scenario(base_graph):
    scorer = CascadeSeverityScorer(high_risk_threshold=0.5)
    
    # Only Web1 is failing highly. DB and API are fine.
    risks = {
        "web1": 0.8,
        "web2": 0.0,
        "api": 0.05,
        "db": 0.0
    }
    
    intel = scorer.aggregate(risks, base_graph)
    
    assert intel["high_risk_node_count"] == 1 # Only web1
    # web1 has no upstream dependents in our graph (no one depends on web1).
    # So cascade size is just 1 (web1 itself).
    assert intel["predicted_cascade_size"] == 1
    assert "web1" in intel["critical_paths"][0]
    
    # Highly localized
    assert intel["risk_concentration_score"] > 0.8
    # System score should be somewhat low but triggers MODERATE because high_risk_node_count == 1
    assert intel["severity_level"] == "MODERATE"

def test_distributed_moderate_risk_scenario(base_graph):
    scorer = CascadeSeverityScorer(high_risk_threshold=0.6) # Tweak threshold slightly
    
    # Risk is spread everywhere but below critical thresholds
    risks = {
        "web1": 0.4,
        "web2": 0.4,
        "api": 0.5,
        "db": 0.5
    }
    
    intel = scorer.aggregate(risks, base_graph)
    
    # Distributed risk -> low concentration score
    # max = 0.5, sum = 1.8. concentration = 0.5 / 1.8 = 0.27
    assert intel["risk_concentration_score"] < 0.35
    
    assert intel["high_risk_node_count"] == 0
    # system_score = (0.4*1 + 0.4*1 + 0.5*1.5 + 0.5*2) / (1+1+1.5+2) = (0.8 + 0.75 + 1.0) / 5.5 = 2.55/5.5 = ~0.46
    assert intel["system_risk_score"] > 0.4
    
    # System score >= 0.4 -> HIGH severity based on rules
    assert intel["severity_level"] == "HIGH"

def test_high_upstream_risk_concentration(base_graph):
    scorer = CascadeSeverityScorer(high_risk_threshold=0.5)
    
    # Database is failing heavily, meaning cascade flows to everyone!
    risks = {
        "web1": 0.5,
        "web2": 0.5,
        "api": 0.7,
        "db": 0.9
    }
    
    intel = scorer.aggregate(risks, base_graph)
    
    # DB, API, Web1, Web2 are all >= 0.5
    assert intel["high_risk_node_count"] == 4
    
    # DB is a high risk root. Dependents of DB is API. Dependents of API are Web1, Web2.
    assert intel["predicted_cascade_size"] == 4
    
    # The longest path should be DB -> API -> Web1 (or DB -> API -> Web2)
    # Length of 3.
    assert len(intel["critical_paths"][0]) == 3
    assert intel["critical_paths"][0][0] == "db"
    
    # Since predicted_cascade_size >= 4 and high_risk_node_count >= 3
    assert intel["severity_level"] == "CRITICAL"
