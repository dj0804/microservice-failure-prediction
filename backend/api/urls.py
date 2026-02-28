from django.urls import path
from .views import (
    InitializeGraphView, InjectFaultView, RunSimulationView, 
    SystemRiskView, EvaluationReportView, GraphStateView,
    RunBenchmarkView
)

urlpatterns = [
    path('initialize-graph', InitializeGraphView.as_view(), name='initialize-graph'),
    path('inject-fault', InjectFaultView.as_view(), name='inject-fault'),
    path('run-simulation', RunSimulationView.as_view(), name='run-simulation'),
    path('system-risk', SystemRiskView.as_view(), name='system-risk'),
    path('evaluation-report', EvaluationReportView.as_view(), name='evaluation-report'),
    path('graph-state', GraphStateView.as_view(), name='graph-state'),
    path('run-benchmark', RunBenchmarkView.as_view(), name='run-benchmark'),
]
