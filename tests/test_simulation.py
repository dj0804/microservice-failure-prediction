import pytest
from src.simulation.models import ServiceNode, DependencyEdge, MetricTick
from src.simulation.graph_engine import GraphEngine
from src.simulation.metrics_generator import MetricsGenerator, FaultEvent
from src.simulation.propagation import RiskPropagationEngine

def test_topology_integrity():
    engine = GraphEngine()
    
    # Topology: Web -> API -> Auth -> DB
    nodes = [
        ServiceNode(id="web", service_name="Web Frontend"),
        ServiceNode(id="api", service_name="API Gateway"),
        ServiceNode(id="auth", service_name="Auth Service"),
        ServiceNode(id="db", service_name="Database")
    ]
    edges = [
        DependencyEdge(source_node_id="web", target_node_id="api", amplification_factor=1.0),
        DependencyEdge(source_node_id="api", target_node_id="auth", amplification_factor=1.5),
        DependencyEdge(source_node_id="api", target_node_id="db", amplification_factor=2.0)
    ]
    
    engine.build_from_definitions(nodes, edges)
    
    assert len(engine.get_all_nodes()) == 4
    assert len(engine.get_upstream_dependents("api")) == 1 # Web depends on API
    assert "web" in engine.get_upstream_dependents("api")
    assert len(engine.get_downstream_dependencies("api")) == 2 # API depends on Auth, DB

def test_risk_propagation():
    engine = GraphEngine()
    
    # Topology: Web -> API -> DB (Linear)
    nodes = [
        ServiceNode(id="web", service_name="Web", criticality_score=1.0),
        ServiceNode(id="api", service_name="API", criticality_score=1.5),
        ServiceNode(id="db", service_name="DB", criticality_score=2.0)
    ]
    edges = [
        DependencyEdge(source_node_id="web", target_node_id="api", amplification_factor=1.0),
        DependencyEdge(source_node_id="api", target_node_id="db", amplification_factor=2.0)
    ]
    
    engine.build_from_definitions(nodes, edges)
    
    # Force localized risk by directly setting metrics
    db_node = engine.get_node("db")
    db_node.current_metrics = MetricTick(tick_id=1, node_id="db", cpu_utilization=0.9, latency_ms=500, error_rate=0.1)
    engine.update_node(db_node)
    
    risk_engine = RiskPropagationEngine(hop_decay_factor=0.5)
    risks = risk_engine.propagate_risk(engine)
    
    # DB Base Risk Should be > 0. 
    # db_node risk roughly: cpu(0.2) + lat(0.3) + err(0.2) = 0.7. base_risk = 0.7 * 2.0 = 1.4 -> but wait! 
    # my code bounds it between 0 and 1 BEFORE applying criticality score.
    # risk = 0.2 + 0.3 + 0.2 = 0.7 -> bound to 0.7 -> 0.7 * 2.0 = 1.4
    
    assert risks["db"] > 0.0
    
    # Diffused up to API: 
    # api_risk = db_risk * db_amp_factor (which is edge api->db amp=2.0) * hop_decay (0.5)
    # api_risk = 1.4 * 2.0 * 0.5 = 1.4
    assert risks["api"] > 0.0
    
    # Diffused up to Web:
    # web_risk = (next iteration from api) = api_risk * web_amp (edge web->api amp=1.0) * hop_decay (0.5)
    # web_risk = 1.4 * 1.0 * 0.5 = 0.7
    assert risks["web"] > 0.0
    
    # Order of magnitude should diminish further away
    assert risks["db"] == risks["api"] # In this specific case, amp=2 and decay=0.5 mutually cancel
    assert risks["web"] < risks["api"]

def test_metric_fault_injection():
    engine = GraphEngine()
    nodes = [ServiceNode(id="test", service_name="Test")]
    engine.build_from_definitions(nodes, [])
    
    metrics_gen = MetricsGenerator()
    
    # Tick 1: Healthy
    metrics1 = metrics_gen.generate_metrics(tick_id=1, nodes=engine.get_all_nodes())
    assert metrics1["test"].latency_ms < 100
    
    # Inject fault!
    metrics_gen.inject_fault(FaultEvent(node_id="test", fault_type="latency_spike", magnitude=500.0, duration_ticks=2))
    
    # Tick 2: Faulty
    metrics2 = metrics_gen.generate_metrics(tick_id=2, nodes=engine.get_all_nodes())
    assert metrics2["test"].latency_ms >= 500
    
    # Tick 3: Faulty (dur=2)
    metrics3 = metrics_gen.generate_metrics(tick_id=3, nodes=engine.get_all_nodes())
    assert metrics3["test"].latency_ms >= 500
    
    # Tick 4: Healthy again
    metrics4 = metrics_gen.generate_metrics(tick_id=4, nodes=engine.get_all_nodes())
    assert metrics4["test"].latency_ms < 100
