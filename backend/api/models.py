from django.db import models

class ServiceNode(models.Model):
    # We use CharField for ID so it directly maps to the NetworkX node identifiers
    node_id = models.CharField(max_length=255, primary_key=True)
    name = models.CharField(max_length=255)
    criticality = models.FloatField(default=1.0)
    
    def __str__(self):
        return f"{self.name} ({self.node_id})"

class DependencyEdge(models.Model):
    source = models.ForeignKey(ServiceNode, related_name='outgoing_edges', on_delete=models.CASCADE)
    target = models.ForeignKey(ServiceNode, related_name='incoming_edges', on_delete=models.CASCADE)
    amplification = models.FloatField(default=1.0)
    
    class Meta:
        unique_together = ('source', 'target')

class SimulationRun(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    target_node = models.ForeignKey(ServiceNode, null=True, blank=True, on_delete=models.SET_NULL)
    fault_type = models.CharField(max_length=100)
    duration = models.IntegerField(default=1)
    
    # Outcome tracking
    final_severity = models.CharField(max_length=50, default='LOW')
    cascade_size = models.IntegerField(default=0)
    max_system_risk = models.FloatField(default=0.0)

class EvaluationReport(models.Model):
    simulation_run = models.OneToOneField(SimulationRun, on_delete=models.CASCADE, related_name='evaluation')
    cascade_size_error = models.IntegerField(default=0)
    severity_match = models.BooleanField(default=True)
    precision = models.FloatField(default=0.0)
    recall = models.FloatField(default=0.0)
    binary_accuracy = models.BooleanField(default=True)
