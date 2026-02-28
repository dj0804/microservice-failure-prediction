import pytest
from simulation.feedback_learning import DeterministicFeedbackLoop

@pytest.fixture
def feedback():
    return DeterministicFeedbackLoop(
        init_high_risk_threshold=0.5,
        init_amplification_multiplier=1.0,
        init_hop_decay=0.8
    )

def test_rule_4_stable_condition(feedback):
    report = {
        "predicted": {"predicted_cascade_size": 3},
        "actual": {"actual_cascade_size": 3},
        "evaluation": {
            "recall": 0.8,
            "precision": 0.8,
            "cascade_size_error": 0
        }
    }
    res = feedback.update_parameters(report)
    assert "Rule 4: Stability Condition" in res["applied_rules"]
    assert res["updated_parameters"]["high_risk_threshold"] == 0.5
    assert res["updated_parameters"]["amplification_multiplier"] == 1.0

def test_rule_1_persistent_low_recall(feedback):
    # Needs 2 occurrences in window of 3
    report = {
        "predicted": {"predicted_cascade_size": 1},
        "actual": {"actual_cascade_size": 3},
        "evaluation": {
            "recall": 0.33,
            "precision": 1.0,
            "cascade_size_error": 2
        }
    }
    res1 = feedback.update_parameters(report) # 1st occurrence
    assert "Rule 1: Underprediction (Low Recall)" not in res1["applied_rules"]
    
    res2 = feedback.update_parameters(report) # 2nd occurrence triggers pattern
    assert "Rule 1: Underprediction (Low Recall)" in res2["applied_rules"]
    
    assert res2["updated_parameters"]["amplification_multiplier"] == 1.05
    assert res2["updated_parameters"]["high_risk_threshold"] == 0.48
    
    # Push completely to bounds bounds
    for _ in range(25):
        res = feedback.update_parameters(report)
        
    assert res["updated_parameters"]["amplification_multiplier"] == 1.5
    assert res["updated_parameters"]["high_risk_threshold"] == 0.4
    
def test_rule_2_persistent_low_precision(feedback):
    report = {
        "predicted": {"predicted_cascade_size": 3},
        "actual": {"actual_cascade_size": 1},
        "evaluation": {
            "recall": 1.0,
            "precision": 0.33,
            "cascade_size_error": 2
        }
    }
    feedback.update_parameters(report)
    res = feedback.update_parameters(report)
    
    assert "Rule 2: Overprediction (Low Precision)" in res["applied_rules"]
    assert res["updated_parameters"]["high_risk_threshold"] == 0.53
    
    # Push to bounds
    for _ in range(25):
        res = feedback.update_parameters(report)
        
    assert res["updated_parameters"]["high_risk_threshold"] == 0.9

def test_rule_3_underpredict_size(feedback):
    report = {
        "predicted": {"predicted_cascade_size": 1},
        "actual": {"actual_cascade_size": 5},
        "evaluation": {
            "recall": 0.8,
            "precision": 0.8,
            "cascade_size_error": 4
        }
    }
    feedback.update_parameters(report)
    res = feedback.update_parameters(report)
    
    # Underpredicting means risk decays too fast -> Retain more risk -> increase multiplier
    assert res["updated_parameters"]["hop_decay"] == 0.82
    assert "Rule 3: Large Cascade Size Error (Underpredicting -> Reduce Decay)" in res["applied_rules"]

def test_rule_3_overpredict_size(feedback):
    report = {
        "predicted": {"predicted_cascade_size": 5},
        "actual": {"actual_cascade_size": 1},
        "evaluation": {
            "recall": 0.8,
            "precision": 0.8,
            "cascade_size_error": 4
        }
    }
    feedback.update_parameters(report)
    res = feedback.update_parameters(report)
    
    # Overpredict -> Risk travels too far -> increase decay / decrease multiplier
    assert res["updated_parameters"]["hop_decay"] == 0.78
    assert "Rule 3: Large Cascade Size Error (Overpredicting -> Increase Decay)" in res["applied_rules"]

def test_demonstration_5_cycles(feedback):
    # Simulates continuous feedback over 5 sequential ticks showing stabilization attempts
    
    # Tick 1 & 2: Extreme underprediction (low recall) -> Should trigger Rule 1 at Tick 2
    report_low_recall = {
        "predicted": {"predicted_cascade_size": 1},
        "actual": {"actual_cascade_size": 5},
        "evaluation": {"recall": 0.4, "precision": 1.0, "cascade_size_error": 4}
    }
    feedback.update_parameters(report_low_recall)
    res2 = feedback.update_parameters(report_low_recall)
    
    assert "Rule 1: Underprediction (Low Recall)" in res2["applied_rules"]
    assert "Rule 3: Large Cascade Size Error (Underpredicting -> Reduce Decay)" in res2["applied_rules"]
    # amp goes 1.0 -> 1.05. hop_decay goes 0.8 -> 0.82
    assert res2["updated_parameters"]["amplification_multiplier"] == 1.05
    assert res2["updated_parameters"]["hop_decay"] == 0.82
    
    # Tick 3 & 4: System over-corrects -> Low precision, high cascade size error (Overpredict)
    report_low_precision = {
        "predicted": {"predicted_cascade_size": 6},
        "actual": {"actual_cascade_size": 2},
        "evaluation": {"recall": 1.0, "precision": 0.3, "cascade_size_error": 4}
    }
    res3 = feedback.update_parameters(report_low_precision)
    # The history window is now [report_low_recall, report_low_precision].
    # Both Rules need count >= 2. low_recall_count=1, low_precision=1. No rule 1 or 2 right now.
    
    res4 = feedback.update_parameters(report_low_precision)
    # History: [report_low_precision, report_low_precision].
    # low_precision_count=2 -> Rule 2 triggers. cascade size error is large -> Rule 3 overpredict triggers.
    assert "Rule 2: Overprediction (Low Precision)" in res4["applied_rules"]
    assert "Rule 3: Large Cascade Size Error (Overpredicting -> Increase Decay)" in res4["applied_rules"]
    # hop_decay decreases 0.84 -> 0.82. high_risk_threshold increases +0.03
    assert res4["updated_parameters"]["hop_decay"] == 0.82
    assert res4["updated_parameters"]["high_risk_threshold"] == 0.51
    
    # Tick 5: System is now perfectly tuned, stable
    report_stable = {
        "predicted": {"predicted_cascade_size": 3},
        "actual": {"actual_cascade_size": 3},
        "evaluation": {"recall": 0.85, "precision": 0.9, "cascade_size_error": 0}
    }
    res5 = feedback.update_parameters(report_stable)
    assert "Rule 4: Stability Condition" in res5["applied_rules"]
    
    # Ensure iteration ticked
    assert res5["iteration"] == 5
