from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .serializers import GraphInitSerializer, FaultInjectionSerializer
from .services import SimulationService
from simulation.benchmark_runner import BenchmarkRunner

class RunBenchmarkView(APIView):
    def post(self, request):
        config = {
            "num_services": request.data.get("num_services", 12),
            "density": request.data.get("density", 0.4),
            "fault_type": request.data.get("fault_type", "cpu_spike"),
            "fault_duration": request.data.get("fault_duration", 4),
            "learning_cycles": request.data.get("learning_cycles", 10),
            "repeated_trials": request.data.get("repeated_trials", 2)
        }
        try:
            runner = BenchmarkRunner(config)
            results = runner.run_benchmark()
            return Response(results, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Ensure we use the Singleton pattern cleanly
sim_service = SimulationService()

class InitializeGraphView(APIView):
    def post(self, request):
        if 'num_services' in request.data:
            num = int(request.data.get('num_services', 10))
            density = float(request.data.get('density', 0.3))
            res = sim_service.initialize_synthetic_graph(num, density)
            return Response(res, status=status.HTTP_201_CREATED)
            
        serializer = GraphInitSerializer(data=request.data)
        if serializer.is_valid():
            try:
                res = sim_service.initialize_graph(
                    serializer.validated_data['nodes'],
                    serializer.validated_data['edges']
                )
                return Response(res, status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class InjectFaultView(APIView):
    def post(self, request):
        serializer = FaultInjectionSerializer(data=request.data)
        if serializer.is_valid():
            res = sim_service.inject_fault(
                serializer.validated_data['target_node_id'],
                serializer.validated_data['fault_type'],
                serializer.validated_data['magnitude'],
                serializer.validated_data['duration_ticks']
            )
            return Response(res, status=status.HTTP_202_ACCEPTED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class RunSimulationView(APIView):
    def post(self, request):
        res = sim_service.run_simulation()
        return Response(res, status=status.HTTP_200_OK)

class SystemRiskView(APIView):
    def get(self, request):
        return Response(sim_service.get_system_risk(), status=status.HTTP_200_OK)

class EvaluationReportView(APIView):
    def get(self, request):
        return Response(sim_service.get_evaluation_report(), status=status.HTTP_200_OK)

class GraphStateView(APIView):
    def get(self, request):
        return Response(sim_service.get_graph_state(), status=status.HTTP_200_OK)
