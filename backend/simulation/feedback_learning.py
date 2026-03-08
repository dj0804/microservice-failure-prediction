from typing import Dict, Any, List
import copy

class DeterministicFeedbackLoop:
    """
    Introduces a pure rule-based, deterministic closed feedback loop to adapt
    parameters (severity threshold, amplification multiplier, hop decay) 
    based on historical prediction evaluation metrics without ML models.
    """
    def __init__(self, 
                 init_high_risk_threshold: float = 0.5, 
                 init_amplification_multiplier: float = 1.0, 
                 init_hop_decay: float = 0.8):
        self._init_high_risk_threshold = init_high_risk_threshold
        self._init_amplification_multiplier = init_amplification_multiplier
        self._init_hop_decay = init_hop_decay
        
        self.registry = {
            "high_risk_threshold": init_high_risk_threshold,
            "amplification_multiplier": init_amplification_multiplier, # Global booster/reducer to edge amp
            "hop_decay": init_hop_decay, # Flow multiplier (0.8 = decay by 20% per hop)
            "learning_iteration": 0
        }
        self.history: List[Dict[str, Any]] = []

    def get_current_parameters(self) -> Dict[str, Any]:
        return copy.deepcopy(self.registry)
        
    def reset_parameters(self):
        self.registry = {
            "high_risk_threshold": self._init_high_risk_threshold,
            "amplification_multiplier": self._init_amplification_multiplier,
            "hop_decay": self._init_hop_decay,
            "learning_iteration": 0
        }
        self.history.clear()

    def update_parameters(self, evaluation_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingests the latest Counterfactual Evaluation report and adjusts the
        tunable parameters incrementally if persistent errors are detected across 
        the sliding window of 3 events.
        """
        self.history.append(evaluation_report)
        if len(self.history) > 3:
            self.history.pop(0)

        prev_registry = copy.deepcopy(self.registry)
        applied_rules = []
        
        eval_metrics = evaluation_report.get("evaluation", {})
        recall = eval_metrics.get("recall", 1.0)
        precision = eval_metrics.get("precision", 1.0)
        cascade_size_error_abs = eval_metrics.get("cascade_size_error", 0)
        
        predicted_size = evaluation_report.get("predicted", {}).get("predicted_cascade_size", 0)
        actual_size = evaluation_report.get("actual", {}).get("actual_cascade_size", 0)
        
        # Rule 4: Stability
        if precision >= 0.75 and recall >= 0.75 and cascade_size_error_abs <= 1:
            applied_rules.append("Rule 4: Stability Condition")
            self.registry["learning_iteration"] += 1
            return {
                "iteration": self.registry["learning_iteration"],
                "previous_parameters": prev_registry,
                "updated_parameters": copy.deepcopy(self.registry),
                "applied_rules": applied_rules
            }

        # Analyze History for Anti-Oscillation requirement
        low_recall_count = sum(1 for h in self.history if h.get("evaluation", {}).get("recall", 1.0) < 0.6)
        low_precision_count = sum(1 for h in self.history if h.get("evaluation", {}).get("precision", 1.0) < 0.6)
        
        # Consistent directional errors for Rule 3
        underpredict_count = sum(1 for h in self.history if h.get("actual", {}).get("actual_cascade_size", 0) > h.get("predicted", {}).get("predicted_cascade_size", 0) and h.get("evaluation", {}).get("cascade_size_error", 0) > 2)
        overpredict_count = sum(1 for h in self.history if h.get("predicted", {}).get("predicted_cascade_size", 0) > h.get("actual", {}).get("actual_cascade_size", 0) and h.get("evaluation", {}).get("cascade_size_error", 0) > 2)

        # Rule 1: Underprediction (Low Recall)
        if recall < 0.6 and low_recall_count >= 2:
            self.registry["amplification_multiplier"] += 0.05
            self.registry["high_risk_threshold"] -= 0.02
            
            if self.registry["amplification_multiplier"] > 1.5:
                self.registry["amplification_multiplier"] = 1.5
            if self.registry["high_risk_threshold"] < 0.4:
                self.registry["high_risk_threshold"] = 0.4
                
            applied_rules.append("Rule 1: Underprediction (Low Recall)")

        # Rule 2: Overprediction (Low Precision)
        elif precision < 0.6 and low_precision_count >= 2:
            self.registry["high_risk_threshold"] += 0.03
            
            if self.registry["high_risk_threshold"] > 0.9:
                self.registry["high_risk_threshold"] = 0.9
                
            applied_rules.append("Rule 2: Overprediction (Low Precision)")

        # Rule 3: Large Cascade Size Error
        if cascade_size_error_abs > 2:
            if underpredict_count >= 2:
                # Underpredicting size means we lose risk too fast. We need to reduce the decay.
                # Since hop_decay is a multiplier (e.g. 0.8 retains 80%), reducing decay means increasing the multiplier (e.g. 0.82)
                self.registry["hop_decay"] += 0.02
                applied_rules.append("Rule 3: Large Cascade Size Error (Underpredicting -> Reduce Decay)")
            elif overpredict_count >= 2:
                # Overpredicting means risk travels too far. We need to increase decay (decrease multiplier).
                self.registry["hop_decay"] -= 0.02
                applied_rules.append("Rule 3: Large Cascade Size Error (Overpredicting -> Increase Decay)")
                
            if self.registry["hop_decay"] < 0.4:
                self.registry["hop_decay"] = 0.4
            elif self.registry["hop_decay"] > 0.9:
                self.registry["hop_decay"] = 0.9
                
        # Clean floating points
        self.registry["amplification_multiplier"] = round(self.registry["amplification_multiplier"], 3)
        self.registry["high_risk_threshold"] = round(self.registry["high_risk_threshold"], 3)
        self.registry["hop_decay"] = round(self.registry["hop_decay"], 3)
            
        self.registry["learning_iteration"] += 1
        
        return {
            "iteration": self.registry["learning_iteration"],
            "previous_parameters": prev_registry,
            "updated_parameters": copy.deepcopy(self.registry),
            "applied_rules": applied_rules
        }
