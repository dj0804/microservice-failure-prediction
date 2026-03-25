"""
Tests for the GraphFailurePredictor (Graph Learning Agent).

Covers:
1. Feature extraction
2. Graph-aware embedding (weighted neighbor aggregation)
3. Cold-start heuristic prediction
4. Training loop (train_on_batch + update_model)
5. Post-training learned prediction
6. Output format validation
7. Model persistence (save/load)
8. Temporal signals (delta_cpu)
9. Integration: end-to-end predict → train → predict
"""
import os
import tempfile
import pytest
import numpy as np

from simulation.models import ServiceNode, DependencyEdge, MetricTick
from simulation.graph_engine import GraphEngine
from simulation.graph_learning_model import GraphFailurePredictor


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def linear_graph():
    """Topology: Web -> API -> DB (linear chain)"""
    engine = GraphEngine()
    nodes = [
        ServiceNode(id="web", service_name="Web", criticality_score=1.0),
        ServiceNode(id="api", service_name="API", criticality_score=1.5),
        ServiceNode(id="db", service_name="DB", criticality_score=2.0),
    ]
    edges = [
        DependencyEdge(source_node_id="web", target_node_id="api", amplification_factor=1.0),
        DependencyEdge(source_node_id="api", target_node_id="db", amplification_factor=2.0),
    ]
    engine.build_from_definitions(nodes, edges)
    return engine


@pytest.fixture
def fan_graph():
    """
    Topology:
        Web1 \\
              -> API -> DB
        Web2 /
    """
    engine = GraphEngine()
    nodes = [
        ServiceNode(id="web1", service_name="Web 1", criticality_score=1.0),
        ServiceNode(id="web2", service_name="Web 2", criticality_score=1.0),
        ServiceNode(id="api", service_name="API Gateway", criticality_score=1.5),
        ServiceNode(id="db", service_name="Database", criticality_score=2.0),
    ]
    edges = [
        DependencyEdge(source_node_id="web1", target_node_id="api", amplification_factor=1.0),
        DependencyEdge(source_node_id="web2", target_node_id="api", amplification_factor=1.2),
        DependencyEdge(source_node_id="api", target_node_id="db", amplification_factor=2.0),
    ]
    engine.build_from_definitions(nodes, edges)
    return engine


@pytest.fixture
def predictor():
    return GraphFailurePredictor(high_risk_threshold=0.7)


def _set_metrics(engine, node_id, cpu, latency_ms, error_rate, tick_id=1):
    """Helper to set metrics on a node."""
    node = engine.get_node(node_id)
    node.current_metrics = MetricTick(
        tick_id=tick_id, node_id=node_id,
        cpu_utilization=cpu, latency_ms=latency_ms, error_rate=error_rate
    )
    engine.update_node(node)


# =========================================================================
# 1. Feature Extraction
# =========================================================================

class TestFeatureExtraction:
    def test_extracts_correct_feature_count(self, predictor, linear_graph):
        _set_metrics(linear_graph, "db", cpu=0.9, latency_ms=500, error_rate=0.1)
        nodes = linear_graph.get_all_nodes()
        features = predictor._extract_node_features(nodes, linear_graph)

        assert len(features) == 3
        for feat in features.values():
            assert len(feat) == 6  # cpu, latency_norm, error_rate, criticality, degree, dep_count

    def test_feature_values_match_metrics(self, predictor, linear_graph):
        _set_metrics(linear_graph, "db", cpu=0.9, latency_ms=500, error_rate=0.1)
        nodes = linear_graph.get_all_nodes()
        features = predictor._extract_node_features(nodes, linear_graph)

        db_feat = features["db"]
        assert db_feat[0] == pytest.approx(0.9)  # cpu
        assert 0 < db_feat[1] < 1  # latency_norm (log-scaled)
        assert db_feat[2] == pytest.approx(0.1)  # error_rate
        assert db_feat[3] == pytest.approx(2.0)  # criticality_score

    def test_no_metrics_yields_zeros(self, predictor, linear_graph):
        # No metrics set — should default to zeros
        nodes = linear_graph.get_all_nodes()
        features = predictor._extract_node_features(nodes, linear_graph)
        for feat in features.values():
            assert feat[0] == 0.0  # cpu
            assert feat[1] == 0.0  # latency
            assert feat[2] == 0.0  # error_rate


# =========================================================================
# 2. Graph-Aware Embedding
# =========================================================================

class TestGraphEmbeddings:
    def test_embedding_dimension(self, predictor, linear_graph):
        _set_metrics(linear_graph, "db", cpu=0.9, latency_ms=500, error_rate=0.1)
        nodes = linear_graph.get_all_nodes()
        raw_features = predictor._extract_node_features(nodes, linear_graph)
        adjacency, edge_weights = predictor._build_adjacency(linear_graph)
        embeddings = predictor._compute_graph_embeddings(nodes, raw_features, adjacency, edge_weights)

        assert embeddings.shape == (3, 12)  # 6 node features + 6 neighbor-aggregated

    def test_neighbor_aggregation_uses_weights(self, predictor, fan_graph):
        _set_metrics(fan_graph, "web1", cpu=0.5, latency_ms=100, error_rate=0.02)
        _set_metrics(fan_graph, "web2", cpu=0.3, latency_ms=80, error_rate=0.01)
        _set_metrics(fan_graph, "api", cpu=0.6, latency_ms=150, error_rate=0.03)
        _set_metrics(fan_graph, "db", cpu=0.9, latency_ms=500, error_rate=0.1)

        nodes = fan_graph.get_all_nodes()
        raw_features = predictor._extract_node_features(nodes, fan_graph)
        adjacency, edge_weights = predictor._build_adjacency(fan_graph)
        embeddings = predictor._compute_graph_embeddings(nodes, raw_features, adjacency, edge_weights)

        # API has 3 neighbors: web1, web2, db. Neighbor part should be non-zero.
        api_idx = [i for i, n in enumerate(nodes) if n.id == "api"][0]
        neighbor_part = embeddings[api_idx, 6:]
        assert np.any(neighbor_part > 0), "API neighbor aggregation should be non-zero"

    def test_isolated_node_neighbor_zeros(self, predictor):
        """A node with no edges should have zero neighbor embedding."""
        engine = GraphEngine()
        engine.build_from_definitions(
            [ServiceNode(id="solo", service_name="Solo", criticality_score=1.0)], []
        )
        _set_metrics(engine, "solo", cpu=0.5, latency_ms=100, error_rate=0.01)

        nodes = engine.get_all_nodes()
        raw_features = predictor._extract_node_features(nodes, engine)
        adjacency, edge_weights = predictor._build_adjacency(engine)
        embeddings = predictor._compute_graph_embeddings(nodes, raw_features, adjacency, edge_weights)

        neighbor_part = embeddings[0, 6:]
        assert np.allclose(neighbor_part, 0.0)


# =========================================================================
# 3. Cold-Start Heuristic Prediction
# =========================================================================

class TestColdStartPrediction:
    def test_heuristic_returns_valid_probabilities(self, predictor, linear_graph):
        _set_metrics(linear_graph, "db", cpu=0.9, latency_ms=500, error_rate=0.1)
        result = predictor.predict_failure_probabilities(linear_graph)

        assert result["prediction_mode"] == "cold_start"
        for prob in result["node_failure_probabilities"].values():
            assert 0.0 <= prob <= 1.0

    def test_high_risk_node_detected_in_cold_start(self, predictor, linear_graph):
        _set_metrics(linear_graph, "db", cpu=0.95, latency_ms=800, error_rate=0.2)
        result = predictor.predict_failure_probabilities(linear_graph)

        # DB should have high failure probability
        assert result["node_failure_probabilities"]["db"] > 0.5

    def test_healthy_node_low_probability(self, predictor, linear_graph):
        _set_metrics(linear_graph, "web", cpu=0.2, latency_ms=30, error_rate=0.001)
        result = predictor.predict_failure_probabilities(linear_graph)

        assert result["node_failure_probabilities"]["web"] < 0.3

    def test_empty_graph(self, predictor):
        engine = GraphEngine()
        engine.build_from_definitions([], [])
        result = predictor.predict_failure_probabilities(engine)

        assert result["node_failure_probabilities"] == {}
        assert result["system_risk_score"] == 0.0
        assert result["severity_level"] == "LOW"


# =========================================================================
# 4. Output Format
# =========================================================================

class TestOutputFormat:
    def test_output_keys(self, predictor, linear_graph):
        _set_metrics(linear_graph, "db", cpu=0.9, latency_ms=500, error_rate=0.1)
        result = predictor.predict_failure_probabilities(linear_graph)

        assert "node_failure_probabilities" in result
        assert "system_risk_score" in result
        assert "high_risk_nodes" in result
        assert "severity_level" in result
        assert "prediction_mode" in result
        assert "high_risk_node_count" in result
        assert "predicted_cascade_size" in result

    def test_severity_classification(self, predictor, linear_graph):
        # Force a critical scenario
        _set_metrics(linear_graph, "web", cpu=0.95, latency_ms=800, error_rate=0.3)
        _set_metrics(linear_graph, "api", cpu=0.95, latency_ms=800, error_rate=0.3)
        _set_metrics(linear_graph, "db", cpu=0.95, latency_ms=800, error_rate=0.3)
        result = predictor.predict_failure_probabilities(linear_graph)

        assert result["severity_level"] in ["HIGH", "CRITICAL"]


# =========================================================================
# 5. Training Loop
# =========================================================================

class TestTraining:
    def test_train_on_batch_accumulates_data(self, predictor, fan_graph):
        _set_metrics(fan_graph, "web1", cpu=0.5, latency_ms=100, error_rate=0.02)
        _set_metrics(fan_graph, "web2", cpu=0.3, latency_ms=80, error_rate=0.01)
        _set_metrics(fan_graph, "api", cpu=0.6, latency_ms=150, error_rate=0.03)
        _set_metrics(fan_graph, "db", cpu=0.9, latency_ms=500, error_rate=0.1)

        predicted = {"web1": 0.1, "web2": 0.05, "api": 0.3, "db": 0.8}
        actual = {"web1": False, "web2": False, "api": False, "db": True}

        result = predictor.train_on_batch(predicted, actual, fan_graph)
        assert result["samples_collected"] == 4
        assert result["ready_to_train"] is False  # Need 10 minimum

    def test_update_model_insufficient_data(self, predictor, fan_graph):
        result = predictor.update_model()
        assert result["status"] == "insufficient_data"

    def test_train_and_update_succeeds(self, predictor, fan_graph):
        """Accumulate enough data and train successfully."""
        for tick in range(4):
            _set_metrics(fan_graph, "web1", cpu=0.2 + tick * 0.1, latency_ms=50, error_rate=0.01, tick_id=tick)
            _set_metrics(fan_graph, "web2", cpu=0.3, latency_ms=60, error_rate=0.01, tick_id=tick)
            _set_metrics(fan_graph, "api", cpu=0.4 + tick * 0.05, latency_ms=100, error_rate=0.02, tick_id=tick)
            _set_metrics(fan_graph, "db", cpu=0.7 + tick * 0.05, latency_ms=300 + tick * 50, error_rate=0.05 + tick * 0.02, tick_id=tick)

            predicted = {"web1": 0.1, "web2": 0.05, "api": 0.2, "db": 0.7}
            actual = {"web1": False, "web2": False, "api": tick > 2, "db": True}
            predictor.train_on_batch(predicted, actual, fan_graph)

        # Should now have 16 samples (4 nodes × 4 ticks)
        result = predictor.update_model()
        assert result["status"] == "trained"
        assert result["samples_used"] >= 10
        assert result["iteration"] == 1

    def test_post_training_uses_learned_mode(self, predictor, fan_graph):
        """After training, predictions should use the learned model."""
        for tick in range(4):
            for nid in ["web1", "web2", "api", "db"]:
                _set_metrics(fan_graph, nid, cpu=0.5, latency_ms=100, error_rate=0.02, tick_id=tick)
            
            actual = {"web1": False, "web2": False, "api": False, "db": True}
            predictor.train_on_batch({nid: 0.5 for nid in actual}, actual, fan_graph)

        predictor.update_model()

        # Now predict — should use learned model
        result = predictor.predict_failure_probabilities(fan_graph)
        assert result["prediction_mode"] == "learned"
        for prob in result["node_failure_probabilities"].values():
            assert 0.0 <= prob <= 1.0


# =========================================================================
# 6. Model Persistence
# =========================================================================

class TestPersistence:
    def test_save_and_load_model(self, predictor, fan_graph):
        # Train the model
        for tick in range(4):
            for nid in ["web1", "web2", "api", "db"]:
                _set_metrics(fan_graph, nid, cpu=0.5, latency_ms=100, error_rate=0.02, tick_id=tick)
            actual = {"web1": False, "web2": False, "api": False, "db": True}
            predictor.train_on_batch({nid: 0.5 for nid in actual}, actual, fan_graph)

        predictor.update_model()

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            predictor.save_model(path)
            assert os.path.exists(path)

            # Load into a fresh predictor
            new_predictor = GraphFailurePredictor()
            assert new_predictor.load_model(path) is True
            assert new_predictor._is_trained is True
            assert new_predictor._training_iteration == 1

            # Predictions should work
            result = new_predictor.predict_failure_probabilities(fan_graph)
            assert result["prediction_mode"] == "learned"

    def test_load_nonexistent_returns_false(self, predictor):
        assert predictor.load_model("/nonexistent/path/model.joblib") is False


# =========================================================================
# 7. Temporal Signals
# =========================================================================

class TestTemporalFeatures:
    def test_delta_cpu_computed_across_ticks(self, predictor, linear_graph):
        # Tick 1: CPU = 0.3
        _set_metrics(linear_graph, "db", cpu=0.3, latency_ms=50, error_rate=0.01, tick_id=1)
        predictor.predict_failure_probabilities(linear_graph)

        # Tick 2: CPU = 0.9 — delta should be +0.6
        _set_metrics(linear_graph, "db", cpu=0.9, latency_ms=500, error_rate=0.1, tick_id=2)

        nodes = linear_graph.get_all_nodes()
        raw_features = predictor._extract_node_features(nodes, linear_graph)
        adjacency, edge_weights = predictor._build_adjacency(linear_graph)
        embeddings = predictor._compute_graph_embeddings(nodes, raw_features, adjacency, edge_weights)
        full = predictor._append_temporal_features(nodes, embeddings)

        db_idx = [i for i, n in enumerate(nodes) if n.id == "db"][0]
        # Last 2 features: [prev_cpu, delta_cpu]
        prev_cpu = full[db_idx, -2]
        delta_cpu = full[db_idx, -1]

        assert prev_cpu == pytest.approx(0.3)
        assert delta_cpu == pytest.approx(0.6)


# =========================================================================
# 8. Integration: End-to-End Flow
# =========================================================================

class TestEndToEnd:
    def test_full_predict_train_predict_cycle(self, fan_graph):
        predictor = GraphFailurePredictor(high_risk_threshold=0.7)

        # --- Phase 1: Cold-start predictions ---
        _set_metrics(fan_graph, "web1", cpu=0.2, latency_ms=30, error_rate=0.001)
        _set_metrics(fan_graph, "web2", cpu=0.2, latency_ms=25, error_rate=0.001)
        _set_metrics(fan_graph, "api", cpu=0.3, latency_ms=80, error_rate=0.01)
        _set_metrics(fan_graph, "db", cpu=0.9, latency_ms=600, error_rate=0.15)

        cold_result = predictor.predict_failure_probabilities(fan_graph)
        assert cold_result["prediction_mode"] == "cold_start"
        assert cold_result["node_failure_probabilities"]["db"] > cold_result["node_failure_probabilities"]["web1"]

        # --- Phase 2: Accumulate training data ---
        for tick in range(5):
            for nid in ["web1", "web2", "api", "db"]:
                cpu = 0.9 if nid == "db" else 0.2 + tick * 0.02
                lat = 600 if nid == "db" else 30 + tick * 5
                err = 0.15 if nid == "db" else 0.001
                _set_metrics(fan_graph, nid, cpu=cpu, latency_ms=lat, error_rate=err, tick_id=tick)

            predicted = {n.id: cold_result["node_failure_probabilities"].get(n.id, 0.0) for n in fan_graph.get_all_nodes()}
            actual = {"web1": False, "web2": False, "api": tick >= 4, "db": True}
            predictor.train_on_batch(predicted, actual, fan_graph)

        # --- Phase 3: Train and verify mode switch ---
        train_result = predictor.update_model()
        assert train_result["status"] == "trained"

        learned_result = predictor.predict_failure_probabilities(fan_graph)
        assert learned_result["prediction_mode"] == "learned"

        # Basic sanity: all probabilities valid
        for prob in learned_result["node_failure_probabilities"].values():
            assert 0.0 <= prob <= 1.0

        # System risk should be a valid number
        assert 0.0 <= learned_result["system_risk_score"] <= 1.0
