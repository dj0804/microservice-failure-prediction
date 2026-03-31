"""
===============================================================================
GRAPH LEARNING FAILURE PREDICTOR — Graph Learning Agent
===============================================================================

Patent: "Failure Propagation Prediction Pattern: A Graph-Learning-Based
Proactive Failure Management Mechanism for Cloud Microservices"

This module implements the Graph Learning Agent described in the patent,
replacing the deterministic risk propagation engine with a probabilistic,
graph-informed failure prediction system that learns from past outcomes.

System Flow:
    Graph + Metrics → Graph Learning Model → Failure Probability Prediction
                    → Learned Cascade Behavior → Feedback → Model Training
                    → Improved Predictions

Replaces:
    - RiskPropagationEngine (BFS + hop decay)
    - CascadeSeverityScorer (partially — severity classification retained)
    - DeterministicFeedbackLoop (rule-based parameter tuning)

===============================================================================
"""

import os
import math
import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import networkx as nx

from .graph_engine import GraphEngine
from .models import ServiceNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FEATURE_DIM = 6          # node_features: cpu, latency_norm, error_rate, criticality, degree, dep_count
_EMBEDDING_DIM = _FEATURE_DIM * 2  # concat(node_features, weighted_mean(neighbor_features))
_TEMPORAL_DIM = 2         # delta_cpu, previous_cpu
_TOTAL_INPUT_DIM = _EMBEDDING_DIM + _TEMPORAL_DIM  # 14
_MIN_SAMPLES_FOR_TRAINING = 10
_HIGH_RISK_THRESHOLD = 0.7
_CASCADE_TOP_K_PATHS = 3


class GraphFailurePredictor:
    """
    Graph Learning Agent — probabilistic failure predictor for cloud
    microservice topologies.

    This agent:
        1. Learns node failure probabilities P(failure) ∈ [0, 1]
        2. Incorporates graph structure via weighted neighbor aggregation
           (edge amplification factors as weights)
        3. Improves over time by training on prediction vs actual outcomes
        4. Persists its learned model across simulation runs

    Cold-Start Behavior:
        Before sufficient training data is collected (< MIN_SAMPLES),
        the agent operates in "cold start mode" using a heuristic scoring
        function similar to the legacy calculate_base_risk(). Once enough
        labeled data is accumulated, the agent automatically switches to
        the learned MLP model.

    Training Loop:
        Replaces rule-based feedback (amplification/decay/threshold tuning)
        with model weight updates via binary cross-entropy optimization.
    """

    def __init__(self, high_risk_threshold: float = _HIGH_RISK_THRESHOLD):
        self.high_risk_threshold = high_risk_threshold

        # -- MLP Model (lazy-initialized on first training) --
        self._model = None
        self._is_trained = False

        # -- Training data accumulator --
        self._X_buffer: List[np.ndarray] = []
        self._y_buffer: List[int] = []

        # -- Temporal state: tracks previous tick's CPU per node --
        self._prev_cpu: Dict[str, float] = {}

        # -- Metric History for Dynamic Attention (rolling window of last 10 ticks) --
        self._metric_history: Dict[str, Dict[str, List[float]]] = {}

        # -- Training iteration counter --
        self._training_iteration: int = 0

    # ======================================================================
    # PUBLIC API
    # ======================================================================

    def predict_failure_probabilities(
        self, graph_engine: GraphEngine
    ) -> Dict[str, Any]:
        """
        Core prediction method — replaces propagate_risk() + aggregate().

        For every node in the graph, produces a failure probability
        P(failure) ∈ [0, 1] using either the trained MLP model or a
        heuristic fallback (cold-start mode).

        Returns:
            {
                "node_failure_probabilities": Dict[node_id, float],
                "system_risk_score": float,
                "high_risk_nodes": List[str],
                "predicted_affected_nodes": List[str],
                "cascade_size": int,
                "propagation_paths": List[List[str]],
                "propagation_risk_score": float,
                "system_failure_probability": float,
                "severity_level": str,
                "prediction_mode": "learned" | "cold_start"
            }
        """
        nodes = graph_engine.get_all_nodes()
        if not nodes:
            return {
                "node_failure_probabilities": {},
                "system_risk_score": 0.0,
                "high_risk_nodes": [],
                "predicted_affected_nodes": [],
                "cascade_size": 0,
                "propagation_paths": [],
                "propagation_risk_score": 0.0,
                "system_failure_probability": 0.0,
                "high_risk_node_count": 0,
                "predicted_cascade_size": 0,
                "severity_level": "LOW",
                "prediction_mode": "cold_start",
            }

        # 1. Build adjacency and edge weight maps
        adjacency, edge_weights = self._build_adjacency(graph_engine)

        # 2. Extract per-node raw features
        raw_features = self._extract_node_features(nodes, graph_engine)

        # 3. Compute graph-aware embeddings (neighbor aggregation)
        embeddings = self._compute_graph_embeddings(
            nodes, raw_features, adjacency, edge_weights
        )

        # 4. Append temporal signals
        full_features = self._append_temporal_features(nodes, embeddings)

        # 5. Predict probabilities
        if self._is_trained and self._model is not None:
            probabilities = self._predict_with_model(nodes, full_features)
            mode = "learned"
        else:
            probabilities = self._predict_heuristic(nodes, raw_features)
            mode = "cold_start"

        # 6. Update temporal state for next tick
        self._update_temporal_state(nodes)

        # 7. Aggregate system-level outputs
        return self._build_output(nodes, probabilities, mode, graph_engine)

    def train_on_batch(
        self,
        predicted_probabilities: Dict[str, float],
        actual_failures: Dict[str, bool],
        graph_engine: GraphEngine,
    ) -> Dict[str, Any]:
        """
        Accumulates training data from a prediction-vs-actual comparison.

        Called after the CounterfactualLogger finishes tracking an event,
        providing the predicted failure probabilities and the actual
        node failure outcomes.

        Args:
            predicted_probabilities: {node_id: P(failure)} from prediction
            actual_failures: {node_id: True/False} ground truth labels
            graph_engine: current graph state for feature re-extraction

        Returns:
            {"samples_collected": int, "ready_to_train": bool}
        """
        nodes = graph_engine.get_all_nodes()
        adjacency, edge_weights = self._build_adjacency(graph_engine)
        raw_features = self._extract_node_features(nodes, graph_engine)
        embeddings = self._compute_graph_embeddings(
            nodes, raw_features, adjacency, edge_weights
        )
        full_features = self._append_temporal_features(nodes, embeddings)

        for i, node in enumerate(nodes):
            if node.id in actual_failures:
                self._X_buffer.append(full_features[i])
                self._y_buffer.append(1 if actual_failures[node.id] else 0)

        total_samples = len(self._y_buffer)
        return {
            "samples_collected": total_samples,
            "ready_to_train": total_samples >= _MIN_SAMPLES_FOR_TRAINING,
        }

    def update_model(self) -> Dict[str, Any]:
        """
        Retrains the MLP model on all accumulated training data.

        Uses binary cross-entropy (log_loss) as the loss function.
        After training, the agent switches from cold-start heuristic
        to the learned model for all future predictions.

        Returns:
            {"status": str, "iteration": int, "samples_used": int, "score": float}
        """
        if len(self._y_buffer) < _MIN_SAMPLES_FOR_TRAINING:
            return {
                "status": "insufficient_data",
                "iteration": self._training_iteration,
                "samples_used": len(self._y_buffer),
                "score": 0.0,
            }

        X = np.array(self._X_buffer)
        y = np.array(self._y_buffer)

        # Ensure we have both classes represented; if not, duplicate with noise
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            # Synthesize a minority sample to allow training
            minority_label = 1 if 0 in unique_classes else 0
            synthetic_x = X[0].copy() + np.random.normal(0, 0.01, X.shape[1])
            X = np.vstack([X, synthetic_x.reshape(1, -1)])
            y = np.append(y, minority_label)

        # Lazy-import sklearn to keep module load lightweight
        from sklearn.neural_network import MLPClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold

        # Build or rebuild the base MLP
        base_mlp = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42,
            warm_start=True,
        )

        # Calibrate probabilities for reliable P(failure) outputs
        n_splits = min(3, max(2, len(unique_classes)))
        if len(y) >= n_splits * 2:
            try:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                calibrated = CalibratedClassifierCV(
                    estimator=base_mlp, cv=cv, method="sigmoid"
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    calibrated.fit(X, y)
                self._model = calibrated
            except Exception:
                # Fallback: train without calibration
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    base_mlp.fit(X, y)
                self._model = base_mlp
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base_mlp.fit(X, y)
            self._model = base_mlp

        self._is_trained = True
        self._training_iteration += 1

        # Score on training data
        score = float(self._model.score(X, y))

        return {
            "status": "trained",
            "iteration": self._training_iteration,
            "samples_used": len(y),
            "score": round(score, 4),
        }

    def save_model(self, path: str) -> None:
        """
        Persists the learned model, training buffers, and temporal state
        to disk so that learning carries over across simulation runs.

        Args:
            path: filesystem path for the saved model file (.joblib)
        """
        import joblib

        state = {
            "model": self._model,
            "is_trained": self._is_trained,
            "X_buffer": self._X_buffer,
            "y_buffer": self._y_buffer,
            "prev_cpu": self._prev_cpu,
            "metric_history": self._metric_history,
            "training_iteration": self._training_iteration,
            "high_risk_threshold": self.high_risk_threshold,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(state, path)
        logger.info("Graph Learning Agent model saved to %s", path)

    def load_model(self, path: str) -> bool:
        """
        Loads a previously saved model from disk.

        Args:
            path: filesystem path to the saved model file (.joblib)

        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        if not os.path.exists(path):
            logger.warning("Model file not found at %s — starting fresh", path)
            return False

        import joblib

        try:
            state = joblib.load(path)
            self._model = state["model"]
            self._is_trained = state["is_trained"]
            self._X_buffer = state["X_buffer"]
            self._y_buffer = state["y_buffer"]
            self._prev_cpu = state["prev_cpu"]
            self._metric_history = state.get("metric_history", {})
            self._training_iteration = state["training_iteration"]
            self.high_risk_threshold = state["high_risk_threshold"]
            logger.info(
                "Graph Learning Agent model loaded from %s (iteration %d)",
                path,
                self._training_iteration,
            )
            return True
        except Exception as e:
            logger.error("Failed to load model from %s: %s", path, e)
            return False

    # ======================================================================
    # INTERNAL: FEATURE EXTRACTION
    # ======================================================================

    def _extract_node_features(
        self, nodes: List[ServiceNode], graph_engine: GraphEngine
    ) -> Dict[str, np.ndarray]:
        """
        Extracts a feature vector for each node:
            [cpu_utilization, latency_normalized, error_rate,
             criticality_score, node_degree, dependency_count]

        Latency is normalized via log-scaling to compress the range.
        """
        features = {}
        for node in nodes:
            m = node.current_metrics
            if m:
                cpu = m.cpu_utilization
                latency_norm = min(1.0, math.log1p(m.latency_ms) / math.log1p(1000))
                error_rate = m.error_rate
            else:
                cpu = 0.0
                latency_norm = 0.0
                error_rate = 0.0

            # Structural features
            degree = len(graph_engine.get_upstream_dependents(node.id)) + \
                     len(graph_engine.get_downstream_dependencies(node.id))
            dep_count = len(graph_engine.get_downstream_dependencies(node.id))

            features[node.id] = np.array([
                cpu,
                latency_norm,
                error_rate,
                node.criticality_score,
                float(degree),
                float(dep_count),
            ], dtype=np.float64)

        return features

    def _calculate_dynamic_attention(self, node_a: str, node_b: str, base_weight: float) -> float:
        """
        Calculates a dynamic attention score (edge weight multiplier) between two nodes
        based on the real-time Pearson Correlation of their last 10 telemetry ticks.
        """
        hist_a = self._metric_history.get(node_a)
        hist_b = self._metric_history.get(node_b)
        
        # If we don't have enough history yet, return static amplification factor
        if not hist_a or not hist_b or len(hist_a["cpu"]) < 5:
            return base_weight
            
        # Correlate CPU vectors
        cpu_a, cpu_b = hist_a["cpu"], hist_b["cpu"]
        
        # Handle zero variance perfectly flat arrays which would crash np.corrcoef with NaN
        std_a = np.std(cpu_a)
        std_b = np.std(cpu_b)
        
        correlation = 0.0
        if std_a > 1e-6 and std_b > 1e-6:
            r = np.corrcoef(cpu_a, cpu_b)[0, 1]
            if not np.isnan(r):
                correlation = float(r)
                
        # Dynamic Multiplier: Base * (1 + max(0, correlation))
        # Meaning: High positive sync = up to doubled attention. Uncorrelated = Base weight.
        return float(base_weight * (1.0 + max(0.0, correlation)))

    def _compute_graph_embeddings(
        self,
        nodes: List[ServiceNode],
        raw_features: Dict[str, np.ndarray],
        adjacency: Dict[str, List[str]],
        edge_weights: Dict[Tuple[str, str], float],
    ) -> np.ndarray:
        """
        Graph-aware representation learning.

        For each node, computes:
            embedding = concat(node_features, weighted_mean(neighbor_features))

        Neighbor aggregation uses edge amplification factors as weights,
        replacing explicit BFS propagation with learned structural context.

        Returns:
            np.ndarray of shape (num_nodes, FEATURE_DIM * 2)
        """
        embeddings = []
        for node in nodes:
            node_feat = raw_features[node.id]
            neighbors = adjacency.get(node.id, [])

            if neighbors:
                weighted_feats = []
                total_weight = 0.0
                for nb_id in neighbors:
                    if nb_id in raw_features:
                        w = edge_weights.get((node.id, nb_id), 1.0)
                        
                        # Apply dynamic attention instead of purely static amplification
                        dynamic_w = self._calculate_dynamic_attention(node.id, nb_id, w)
                        
                        weighted_feats.append(raw_features[nb_id] * dynamic_w)
                        total_weight += dynamic_w

                if total_weight > 0:
                    neighbor_agg = np.sum(weighted_feats, axis=0) / total_weight
                else:
                    neighbor_agg = np.zeros(_FEATURE_DIM, dtype=np.float64)
            else:
                neighbor_agg = np.zeros(_FEATURE_DIM, dtype=np.float64)

            embedding = np.concatenate([node_feat, neighbor_agg])
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float64)

    def _append_temporal_features(
        self, nodes: List[ServiceNode], embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Appends temporal signals to each node's embedding:
            - previous_cpu: CPU utilization from the prior tick
            - delta_cpu: change in CPU utilization (current - previous)

        Allows the model to capture evolving failure patterns over time.

        Returns:
            np.ndarray of shape (num_nodes, EMBEDDING_DIM + TEMPORAL_DIM)
        """
        temporal = []
        for node in nodes:
            current_cpu = 0.0
            if node.current_metrics:
                current_cpu = node.current_metrics.cpu_utilization

            prev_cpu = self._prev_cpu.get(node.id, current_cpu)
            delta_cpu = current_cpu - prev_cpu

            temporal.append([prev_cpu, delta_cpu])

        temporal_arr = np.array(temporal, dtype=np.float64)
        return np.hstack([embeddings, temporal_arr])

    def _update_temporal_state(self, nodes: List[ServiceNode]) -> None:
        """Updates the temporal state and metric history (10-tick rolling window) for dynamic attention."""
        for node in nodes:
            if node.current_metrics:
                self._prev_cpu[node.id] = node.current_metrics.cpu_utilization
                
                if node.id not in self._metric_history:
                    self._metric_history[node.id] = {"cpu": [], "latency": []}
                
                m = self._metric_history[node.id]
                m["cpu"].append(node.current_metrics.cpu_utilization)
                # Store normalized latency for potential future multi-variate correlation
                norm_lat = min(1.0, math.log1p(node.current_metrics.latency_ms) / math.log1p(1000))
                m["latency"].append(norm_lat)
                
                if len(m["cpu"]) > 10:
                    m["cpu"].pop(0)
                    m["latency"].pop(0)

    # ======================================================================
    # INTERNAL: PREDICTION
    # ======================================================================

    def _predict_with_model(
        self, nodes: List[ServiceNode], features: np.ndarray
    ) -> Dict[str, float]:
        """
        Uses the trained MLP model to predict P(failure) for each node.

        Returns calibrated probabilities in [0, 1].
        """
        probabilities = {}
        try:
            proba = self._model.predict_proba(features)
            # predict_proba returns [[P(0), P(1)], ...] — we want P(1)
            failure_col = 1 if proba.shape[1] > 1 else 0
            for i, node in enumerate(nodes):
                probabilities[node.id] = float(
                    np.clip(proba[i][failure_col], 0.0, 1.0)
                )
        except Exception as e:
            logger.warning("Model prediction failed, falling back to heuristic: %s", e)
            raw_features = {
                node.id: features[i][:_FEATURE_DIM] for i, node in enumerate(nodes)
            }
            probabilities = self._predict_heuristic(nodes, raw_features)

        return probabilities

    def _predict_heuristic(
        self, nodes: List[ServiceNode], raw_features: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Cold-start heuristic fallback — produces approximate failure
        probabilities using a sigmoid-based scoring function.

        This mirrors the legacy calculate_base_risk() logic but maps
        output to a proper probability via the sigmoid function.

        Used only when the model has not yet been trained.
        """
        probabilities = {}
        for node in nodes:
            feat = raw_features.get(node.id)
            if feat is None or len(feat) < 4:
                probabilities[node.id] = 0.0
                continue

            cpu, latency_norm, error_rate, criticality = feat[0], feat[1], feat[2], feat[3]

            # Raw risk score (similar to old propagation heuristic)
            raw_score = 0.0
            if cpu > 0.8:
                raw_score += (cpu - 0.8) * 2.0
            if latency_norm > 0.5:
                raw_score += (latency_norm - 0.5) * 0.6
            if error_rate > 0.05:
                raw_score += min(0.4, (error_rate - 0.05) * 4.0)

            raw_score *= criticality

            # Map to probability via sigmoid: P = 1 / (1 + exp(-k*(x - x0)))
            # Tuned so that raw_score ~0.5 maps to P ~0.5
            probability = 1.0 / (1.0 + math.exp(-5.0 * (raw_score - 0.5)))
            probabilities[node.id] = round(probability, 6)

        return probabilities

    # ======================================================================
    # INTERNAL: GRAPH UTILITIES
    # ======================================================================

    def _build_adjacency(
        self, graph_engine: GraphEngine
    ) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float]]:
        """
        Builds adjacency list and edge weight map from the GraphEngine.

        Adjacency includes both upstream dependents and downstream
        dependencies (bidirectional neighborhood) for comprehensive
        graph context during embedding computation.

        Edge weights are the amplification_factor values from DependencyEdge.
        """
        adjacency: Dict[str, List[str]] = {}
        edge_weights: Dict[Tuple[str, str], float] = {}

        for node in graph_engine.get_all_nodes():
            neighbors = set()

            # Upstream: nodes that depend on this node
            for dep_id in graph_engine.get_upstream_dependents(node.id):
                neighbors.add(dep_id)
                edge = graph_engine.get_edge(dep_id, node.id)
                if edge:
                    edge_weights[(node.id, dep_id)] = edge.amplification_factor

            # Downstream: nodes this node depends on
            for dep_id in graph_engine.get_downstream_dependencies(node.id):
                neighbors.add(dep_id)
                edge = graph_engine.get_edge(node.id, dep_id)
                if edge:
                    edge_weights[(node.id, dep_id)] = edge.amplification_factor

            adjacency[node.id] = list(neighbors)

        return adjacency, edge_weights

    # ======================================================================
    # INTERNAL: OUTPUT ASSEMBLY
    # ======================================================================

    def predict_cascade(
        self, graph_engine: GraphEngine, node_probabilities: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Derives cascade insights from predicted node failure probabilities.

        This method is intentionally lightweight and deterministic. It uses
        probability thresholds plus graph topology only — no BFS-based
        propagation decay and no deterministic simulation loops.
        """
        node_ids = [node.id for node in graph_engine.get_all_nodes()]
        total_nodes = len(node_ids)
        if total_nodes == 0:
            return {
                "predicted_affected_nodes": [],
                "cascade_size": 0,
                "propagation_paths": [],
                "propagation_risk_score": 0.0,
                "system_failure_probability": 0.0,
            }

        threshold = self.high_risk_threshold

        affected_nodes = sorted(
            [
                node_id
                for node_id in node_ids
                if node_probabilities.get(node_id, 0.0) >= threshold
            ],
            key=lambda node_id: (-node_probabilities.get(node_id, 0.0), node_id),
        )
        affected_set = set(affected_nodes)

        candidate_paths: List[List[str]] = []
        for node_id in affected_nodes:
            direct_dependents = [
                dep
                for dep in graph_engine.get_upstream_dependents(node_id)
                if dep in affected_set
            ]
            for dependent_id in sorted(
                direct_dependents,
                key=lambda dep: (-node_probabilities.get(dep, 0.0), dep),
            ):
                path = [node_id, dependent_id]
                visited = {node_id, dependent_id}
                current = dependent_id

                while True:
                    next_candidates = [
                        dep
                        for dep in graph_engine.get_upstream_dependents(current)
                        if dep in affected_set and dep not in visited
                    ]
                    if not next_candidates:
                        break

                    next_node = sorted(
                        next_candidates,
                        key=lambda dep: (-node_probabilities.get(dep, 0.0), dep),
                    )[0]
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

                candidate_paths.append(path)

        unique_paths: List[List[str]] = []
        seen = set()
        for path in candidate_paths:
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            unique_paths.append(path)

        unique_paths.sort(
            key=lambda path: (
                -sum(node_probabilities.get(node_id, 0.0) for node_id in path),
                -len(path),
                path,
            )
        )
        propagation_paths = unique_paths[:_CASCADE_TOP_K_PATHS]

        propagation_risk_numerator = 0.0
        for node_id in node_ids:
            if not graph_engine.graph.has_node(node_id):
                continue
            reachable_nodes = nx.descendants(graph_engine.graph.reverse(copy=False), node_id)
            propagation_risk_numerator += node_probabilities.get(node_id, 0.0) * len(reachable_nodes)

        propagation_risk_score = propagation_risk_numerator / total_nodes
        system_failure_probability = len(affected_nodes) / total_nodes

        return {
            "predicted_affected_nodes": affected_nodes,
            "cascade_size": len(affected_nodes),
            "propagation_paths": propagation_paths,
            "propagation_risk_score": round(propagation_risk_score, 4),
            "system_failure_probability": round(system_failure_probability, 4),
        }

    def _build_output(
        self,
        nodes: List[ServiceNode],
        probabilities: Dict[str, float],
        mode: str,
        graph_engine: GraphEngine,
    ) -> Dict[str, Any]:
        """
        Assembles the backward-compatible output dictionary.

        System risk = weighted mean of node failure probabilities
                      (weighted by criticality_score).

        Severity classification follows the same thresholds as the legacy
        CascadeSeverityScorer for backward compatibility.
        """
        # System risk: criticality-weighted mean of probabilities
        total_weighted_prob = 0.0
        total_criticality = 0.0
        for node in nodes:
            prob = probabilities.get(node.id, 0.0)
            total_weighted_prob += prob * node.criticality_score
            total_criticality += node.criticality_score

        system_risk = (
            total_weighted_prob / total_criticality if total_criticality > 0 else 0.0
        )

        # High-risk nodes
        high_risk_nodes = [
            nid for nid, prob in probabilities.items()
            if prob >= self.high_risk_threshold
        ]
        high_risk_count = len(high_risk_nodes)

        cascade_result = self.predict_cascade(graph_engine, probabilities)
        cascade_size = cascade_result["cascade_size"]

        # Severity classification (backward-compatible thresholds)
        severity = self._classify_severity(
            system_risk, high_risk_count, cascade_size
        )

        return {
            "node_failure_probabilities": probabilities,
            "system_risk_score": round(system_risk, 4),
            "high_risk_nodes": high_risk_nodes,
            "predicted_affected_nodes": cascade_result["predicted_affected_nodes"],
            "cascade_size": cascade_result["cascade_size"],
            "propagation_paths": cascade_result["propagation_paths"],
            "propagation_risk_score": cascade_result["propagation_risk_score"],
            "system_failure_probability": cascade_result["system_failure_probability"],
            "high_risk_node_count": high_risk_count,
            "predicted_cascade_size": cascade_size,
            "severity_level": severity,
            "prediction_mode": mode,
        }

    @staticmethod
    def _classify_severity(
        system_score: float, high_risk_count: int, cascade_size: int
    ) -> str:
        """
        Rule-based severity classification — retained from legacy
        CascadeSeverityScorer for backward compatibility.
        """
        if system_score >= 0.7 or cascade_size >= 4 or high_risk_count >= 3:
            return "CRITICAL"
        if system_score >= 0.4 or cascade_size >= 2 or high_risk_count >= 2:
            return "HIGH"
        if system_score >= 0.1 or high_risk_count >= 1:
            return "MODERATE"
        return "LOW"
