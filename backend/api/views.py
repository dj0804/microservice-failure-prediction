from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import GraphInitSerializer, FaultInjectionSerializer
from .services import SimulationService

# Ensure we use the Singleton pattern cleanly
sim_service = SimulationService()

class InitializeGraphView(APIView):
    def post(self, request):
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
