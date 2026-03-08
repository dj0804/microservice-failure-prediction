from typing import Optional, Dict
from pydantic import BaseModel, Field

class MetricTick(BaseModel):
    tick_id: int
    node_id: str
    cpu_utilization: float = Field(default=0.0, description="Synthetic CPU utilization (0.0 to 1.0)")
    latency_ms: float = Field(default=0.0, description="Synthetic latency in milliseconds")
    error_rate: float = Field(default=0.0, description="Synthetic error rate (0.0 to 1.0)")

class ServiceNode(BaseModel):
    id: str
    service_name: str
    criticality_score: float = Field(default=1.0, description="Base importance of the service, used as a multiplier for risk")
    current_metrics: Optional[MetricTick] = None
    calculated_risk_score: float = Field(default=0.0, description="Calculated aggregate risk score for the node")

class DependencyEdge(BaseModel):
    source_node_id: str
    target_node_id: str
    amplification_factor: float = Field(default=1.0, description="How heavily errors multiply across this link")
