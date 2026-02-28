import random
from typing import Dict, List, Optional
from .models import ServiceNode, MetricTick

class FaultEvent:
    def __init__(self, node_id: str, fault_type: str, magnitude: float, duration_ticks: int):
        self.node_id = node_id
        self.fault_type = fault_type  # e.g., 'latency_spike', 'cpu_spike', 'error_spike'
        self.magnitude = magnitude    # 0.0 to 1.0 (or ms for latency)
        self.duration_ticks = duration_ticks
        self.ticks_remaining = duration_ticks

class MetricsGenerator:
    """
    Generates synthetic runtime metrics for a simulated tick.
    Maintains active faults and applies them.
    """
    def __init__(self):
        self.active_faults: List[FaultEvent] = []

    def inject_fault(self, fault: FaultEvent):
        self.active_faults.append(fault)

    def generate_metrics(self, tick_id: int, nodes: List[ServiceNode]) -> Dict[str, MetricTick]:
        metrics = {}
        
        # 1. Generate Baseline Metrics
        for node in nodes:
            # Healthy baseline
            metrics[node.id] = MetricTick(
                tick_id=tick_id,
                node_id=node.id,
                cpu_utilization=random.uniform(0.1, 0.4),
                latency_ms=random.uniform(10.0, 50.0),
                error_rate=random.uniform(0.001, 0.01)
            )

        # 2. Apply active faults
        for fault in self.active_faults:
            if fault.ticks_remaining > 0 and fault.node_id in metrics:
                metric = metrics[fault.node_id]
                if fault.fault_type == 'cpu_spike':
                    metric.cpu_utilization = min(1.0, metric.cpu_utilization + fault.magnitude)
                elif fault.fault_type == 'latency_spike':
                    metric.latency_ms += fault.magnitude
                elif fault.fault_type == 'error_spike':
                    metric.error_rate = min(1.0, metric.error_rate + fault.magnitude)
                
                fault.ticks_remaining -= 1

        # 3. Cleanup expired faults
        self.active_faults = [f for f in self.active_faults if f.ticks_remaining > 0]

        return metrics
