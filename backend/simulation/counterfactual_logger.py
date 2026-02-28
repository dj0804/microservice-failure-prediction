from typing import Dict, Any, List

class CounterfactualLogger:
    """
    Captures predicted cascade metrics before action, tracks actual outcomes 
    across simulation ticks, and computes evaluation metrics (precision, recall, etc.)
    for research validation.
    """
    def __init__(self, high_risk_threshold: float = 0.5):
        self.high_risk_threshold = high_risk_threshold
        
        # State tracking per event
        self.current_prediction_snapshot: Dict[str, Any] = {}
        
        # Tracking actual simulation unfolding
        self.actual_failed_nodes: set = set()
        self.actual_max_system_risk: float = 0.0
        self.start_tick: int = 0
        self.is_tracking: bool = False
        self.ticks_to_stabilization: int = 0
        
    def snapshot_prediction(self, tick_id: int, aggregated_intelligence: Dict[str, Any], raw_risks: Dict[str, float]):
        """
        Takes a snapshot of what the engine PREDICTS will happen.
        """
        predicted_high_risk_nodes = {nid for nid, risk in raw_risks.items() if risk >= self.high_risk_threshold}
        
        self.current_prediction_snapshot = {
            "tick_id": tick_id,
            "system_risk_score": aggregated_intelligence.get("system_risk_score", 0.0),
            "high_risk_node_count": aggregated_intelligence.get("high_risk_node_count", 0),
            "predicted_cascade_size": aggregated_intelligence.get("predicted_cascade_size", 0),
            "severity_level": aggregated_intelligence.get("severity_level", "LOW"),
            "critical_paths": aggregated_intelligence.get("critical_paths", []),
            "predicted_failed_nodes": predicted_high_risk_nodes
        }
        
        # Reset tracker for the new event
        self.actual_failed_nodes = set()
        self.actual_max_system_risk = 0.0
        self.start_tick = tick_id
        self.is_tracking = True
        self.ticks_to_stabilization = 0

    def track_actual_tick(self, tick_id: int, current_intelligence: Dict[str, Any], current_raw_risks: Dict[str, float]):
        """
        Called on every simulation tick to observe what ACTUALLY happened.
        """
        if not self.is_tracking:
            return
            
        sys_risk = current_intelligence.get("system_risk_score", 0.0)
        if sys_risk > self.actual_max_system_risk:
            self.actual_max_system_risk = sys_risk
            
        # Record any node that crossed the failure threshold during tracking
        newly_failed = {nid for nid, risk in current_raw_risks.items() if risk >= self.high_risk_threshold}
        self.actual_failed_nodes.update(newly_failed)

        # Check for stabilization (system return to LOW severity)
        if current_intelligence.get("severity_level", "LOW") == "LOW":
            self.ticks_to_stabilization = tick_id - self.start_tick
            self.is_tracking = False

    def get_actual_outcome_summary(self) -> Dict[str, Any]:
        """
        Returns what actually unfolded before stabilization or intervention.
        """
        return {
            "failed_nodes": list(self.actual_failed_nodes),
            "actual_cascade_size": len(self.actual_failed_nodes),
            "time_to_stabilization_ticks": self.ticks_to_stabilization,
            "max_system_risk_observed": self.actual_max_system_risk
        }

    def evaluate(self) -> Dict[str, Any]:
        """
        Compares prediction vs actual and computes accuracy metrics.
        """
        if not self.current_prediction_snapshot:
            return {"error": "No prediction snapshot available to evaluate."}
            
        pred = self.current_prediction_snapshot
        pred_failed_set = pred["predicted_failed_nodes"]
        actual_failed_set = self.actual_failed_nodes
        
        # 1. Cascade Size Error
        cascade_size_error = abs(pred["predicted_cascade_size"] - len(actual_failed_set))
        
        # 2. Precision & Recall for Node-level Failure
        true_positives = len(pred_failed_set.intersection(actual_failed_set))
        false_positives = len(pred_failed_set - actual_failed_set)
        false_negatives = len(actual_failed_set - pred_failed_set)
        
        precision = true_positives / len(pred_failed_set) if pred_failed_set else (1.0 if not actual_failed_set else 0.0)
        recall = true_positives / len(actual_failed_set) if actual_failed_set else (1.0 if not pred_failed_set else 0.0)
        
        # 3. Binary Cascade Accuracy
        # Let's define a "system cascade" as > 1 node failing
        predicted_cascade_binary = pred["predicted_cascade_size"] > 1
        actual_cascade_binary = len(actual_failed_set) > 1
        binary_accuracy = predicted_cascade_binary == actual_cascade_binary
        
        # 4. Severity Match Heuristic
        # Did the max observed risk justify the predicted severity?
        # A simple approximation: CRITICAL means max_sys_risk > 0.7 or large actual cascade.
        # We classify the actual outcome using roughly the same rules as risk_aggregation to check match.
        actual_severity = "LOW"
        if self.actual_max_system_risk >= 0.7 or len(actual_failed_set) >= 4:
            actual_severity = "CRITICAL"
        elif self.actual_max_system_risk >= 0.4 or len(actual_failed_set) >= 2:
            actual_severity = "HIGH"
        elif self.actual_max_system_risk >= 0.1 or len(actual_failed_set) >= 1:
            actual_severity = "MODERATE"
            
        severity_match = pred["severity_level"] == actual_severity
        
        return {
            "predicted": {
                k: v for k, v in pred.items() if k != "predicted_failed_nodes" 
            },
            "actual": self.get_actual_outcome_summary(),
            "evaluation": {
                "cascade_size_error": cascade_size_error,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "binary_accuracy": binary_accuracy,
                "severity_match": severity_match,
                "actual_inferred_severity": actual_severity
            }
        }
