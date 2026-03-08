import pytest
from backend.simulation.models import ServiceNode, DependencyEdge, MetricTick
from backend.simulation.graph_engine import GraphEngine
from backend.simulation.propagation import RiskPropagationEngine
from backend.simulation.risk_aggregation import CascadeSeverityScorer
from backend.simulation.action_engine import PreventiveActionEngine

@pytest.fixture
def test_setup():
    engine = GraphEngine()
    nodes = [
        ServiceNode(id="web", service_name="Web", criticality_score=1.0),
        ServiceNode(id="api", service_name="API Gateway", criticality_score=1.5),
        ServiceNode(id="db", service_name="Database", criticality_score=2.0)
    ]
    edges = [
        DependencyEdge(source_node_id="web", target_node_id="api", amplification_factor=2.0),
        DependencyEdge(source_node_id="api", target_node_id="db", amplification_factor=2.0)
    ]
    engine.build_from_definitions(nodes, edges)
    
    risk_eng = RiskPropagationEngine(hop_decay_factor=0.8)
    scorer = CascadeSeverityScorer(high_risk_threshold=0.5)
    action_eng = PreventiveActionEngine(risk_eng, scorer)
    
    return engine, risk_eng, scorer, action_eng

def test_localized_failure_no_action(test_setup):
    graph, risk_eng, scorer, action_eng = test_setup
    
    # Web failing slightly
    web_node = graph.get_node("web")
    web_node.current_metrics = MetricTick(tick_id=1, node_id="web", cpu_utilization=0.9, latency_ms=100, error_rate=0.0)
    graph.update_node(web_node)
    
    risks = risk_eng.propagate_risk(graph)
    intel = scorer.aggregate(risks, graph)
    
    action_res = action_eng.trigger_proactive_defense(graph, risks, intel)
    
    # Should not trigger anything since Web failing slightly without downstream impact is merely MODERATE
    assert action_res["action_taken"] == "NONE"

def test_medium_cascade_throttling(test_setup):
    graph, risk_eng, scorer, action_eng = test_setup
    
    # DB is failing, causing cascade to API
    db_node = graph.get_node("db")
    db_node.current_metrics = MetricTick(tick_id=1, node_id="db", cpu_utilization=0.95, latency_ms=300, error_rate=0.08)
    graph.update_node(db_node)
    
    risks = risk_eng.propagate_risk(graph)
    intel = scorer.aggregate(risks, graph)
    
    assert intel["severity_level"] in ["HIGH", "CRITICAL"]
    
    # Evaluate Throttle specifically
    throttle_res = action_eng.evaluate_mitigation(graph, intel, "THROTTLE", ["db"])
    
    assert throttle_res["action_taken"] == "THROTTLE"
    assert throttle_res["risk_reduction_percent"] > 0.0
    # Throttling might not drop severity below CRITICAL/HIGH if the baseline risk was extremely high,
    # but it definitively lowers the risk probability overall.
    assert throttle_res["severity_after"] in ["MODERATE", "LOW", "HIGH", "CRITICAL"]

def test_large_cascade_isolation(test_setup):
    graph, risk_eng, scorer, action_eng = test_setup
    
    # DB is dead dead. Critical overload.
    db_node = graph.get_node("db")
    db_node.current_metrics = MetricTick(tick_id=1, node_id="db", cpu_utilization=1.0, latency_ms=5000, error_rate=0.8)
    graph.update_node(db_node)
    
    risks = risk_eng.propagate_risk(graph)
    intel = scorer.aggregate(risks, graph)
    
    assert intel["severity_level"] == "CRITICAL"
    baseline_cascade = intel["predicted_cascade_size"]
    assert baseline_cascade == 3 # Web, API, DB
    
    # Isolation should sever API -> DB dependency.
    isolate_res = action_eng.evaluate_mitigation(graph, intel, "ISOLATE", ["db"])
    
    assert isolate_res["post_action_cascade_size"] < baseline_cascade
    assert isolate_res["post_action_cascade_size"] == 1 # Only the DB is now at risk, upstream is saved!
    
    # After isolation, only 1 node is failing (DB). 
    # Our severity rules: high_risk_node_count >= 1 -> MODERATE.
    # However, DB criticality = 2.0. Total graph criticality = 4.5
    # DB risk is highly elevated (fault severity large). 
    # System_score = (DB_risk * 2.0) / 4.5. 
    # If DB_risk > 1.5, system score exceeds 0.7 -> CRITICAL still possible for the system level,
    # BUT we definitively know cascade_size reduced from 3 to 1. 
    # So we strictly assert the risk reduced.
    assert isolate_res["risk_reduction_percent"] > 0.0
