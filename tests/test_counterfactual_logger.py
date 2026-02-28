import pytest
from src.simulation.counterfactual_logger import CounterfactualLogger

def test_small_localized_fault_scenario():
    logger = CounterfactualLogger(high_risk_threshold=0.5)
    
    # 1. Prediction Snapshot: We predict only Web1 fails.
    raw_risks_pred = {"web1": 0.6, "api": 0.1}
    intel_pred = {
        "system_risk_score": 0.15,
        "high_risk_node_count": 1,
        "predicted_cascade_size": 1,
        "severity_level": "MODERATE",
    }
    logger.snapshot_prediction(tick_id=1, aggregated_intelligence=intel_pred, raw_risks=raw_risks_pred)
    
    # 2. Actual simulation unfolding
    # Tick 2: Actually matched exactly.
    logger.track_actual_tick(tick_id=2, current_intelligence=intel_pred, current_raw_risks=raw_risks_pred)
    
    # Tick 3: Stabilizes
    logger.track_actual_tick(tick_id=3, current_intelligence={"severity_level": "LOW", "system_risk_score": 0.0}, current_raw_risks={"web1": 0.0, "api": 0.0})
    
    # 3. Evaluate
    report = logger.evaluate()
    eval_mets = report["evaluation"]
    
    assert eval_mets["cascade_size_error"] == 0
    assert eval_mets["precision"] == 1.0 # Predicted web1, actual web1
    assert eval_mets["recall"] == 1.0
    assert eval_mets["binary_accuracy"] is True # Pred dict 1 node (False cascade), actual 1 node (False cascade)
    assert eval_mets["severity_match"] is True

def test_medium_propagation_over_prediction():
    logger = CounterfactualLogger(high_risk_threshold=0.5)
    
    # 1. Prediction Snapshot: API fails, predicting it takes down Web1 and Web2.
    raw_risks_pred = {"api": 0.8, "web1": 0.6, "web2": 0.6}
    intel_pred = {
        "system_risk_score": 0.5,
        "predicted_cascade_size": 3,
        "severity_level": "HIGH",
    }
    logger.snapshot_prediction(tick_id=1, aggregated_intelligence=intel_pred, raw_risks=raw_risks_pred)
    
    # 2. Actual Simulation: API failed, but it only took down Web1. Web2 magically survived (e.g. timeout logic saved it).
    raw_risks_actual = {"api": 0.8, "web1": 0.6, "web2": 0.2}
    intel_actual = {
        "system_risk_score": 0.45,
        "severity_level": "HIGH", # Still HIGH because 2 nodes failed
    }
    logger.track_actual_tick(tick_id=2, current_intelligence=intel_actual, current_raw_risks=raw_risks_actual)
    
    # 3. Evaluate
    report = logger.evaluate()
    eval_mets = report["evaluation"]
    
    assert eval_mets["cascade_size_error"] == 1 # Predicted 3, Actual 2
    # Precision: predicted 3, only 2 were true -> 2/3 = 0.6667
    assert eval_mets["precision"] == pytest.approx(0.6667, 0.01)
    # Recall: out of 2 actual failures, we predicted both -> 2/2 = 1.0
    assert eval_mets["recall"] == 1.0
    assert eval_mets["severity_match"] is True

def test_large_cascading_failure_under_prediction():
    logger = CounterfactualLogger(high_risk_threshold=0.5)
    
    # 1. Prediction Snapshot: We under-predicted. Thought only DB and API would fail.
    raw_risks_pred = {"db": 0.9, "api": 0.6, "web1": 0.3}
    intel_pred = {
        "system_risk_score": 0.4,
        "predicted_cascade_size": 2,
        "severity_level": "HIGH",
    }
    logger.snapshot_prediction(tick_id=1, aggregated_intelligence=intel_pred, raw_risks=raw_risks_pred)
    
    # 2. Actual Simulation: DB took down EVERYTHING, system went critical.
    raw_risks_actual = {"db": 0.9, "api": 0.8, "web1": 0.7, "web2": 0.6}
    intel_actual = {
        "system_risk_score": 0.8,
        "severity_level": "CRITICAL",
    }
    logger.track_actual_tick(tick_id=2, current_intelligence=intel_actual, current_raw_risks=raw_risks_actual)
    
    # 3. Evaluate
    report = logger.evaluate()
    eval_mets = report["evaluation"]
    
    assert eval_mets["cascade_size_error"] == 2 # predicted 2, actual 4
    # Precision: predicted 2 (db, api), both failed -> 2/2 = 1.0
    assert eval_mets["precision"] == 1.0
    # Recall: 4 failed, we only predicted 2 of them -> 2/4 = 0.5
    assert eval_mets["recall"] == 0.5
    assert eval_mets["severity_match"] is False # Predicted HIGH, Actual was CRITICAL
