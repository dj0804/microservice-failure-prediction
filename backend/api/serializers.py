from rest_framework import serializers

class NodeSerializer(serializers.Serializer):
    id = serializers.CharField(max_length=255)
    name = serializers.CharField(max_length=255)
    criticality = serializers.FloatField(default=1.0)

class EdgeSerializer(serializers.Serializer):
    source = serializers.CharField(max_length=255)
    target = serializers.CharField(max_length=255)
    amplification = serializers.FloatField(default=1.0)

class GraphInitSerializer(serializers.Serializer):
    nodes = NodeSerializer(many=True)
    edges = EdgeSerializer(many=True)

class FaultInjectionSerializer(serializers.Serializer):
    target_node_id = serializers.CharField(max_length=255)
    fault_type = serializers.CharField(max_length=100)
    magnitude = serializers.FloatField(default=0.8)
    duration_ticks = serializers.IntegerField(default=3)
