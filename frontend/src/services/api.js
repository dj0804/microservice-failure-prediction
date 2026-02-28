import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const simulationApi = {
    initializeGraph: async (numServices = 10, density = 0.3) => {
        const response = await api.post('/initialize-graph', {
            num_services: numServices,
            density: density
        });
        return response.data;
    },

    getGraphState: async () => {
        const response = await api.get('/graph-state');
        return response.data;
    },

    injectFault: async (nodeId, faultType = 'cpu_spike', magnitude = 1.0, duration = 3) => {
        const response = await api.post('/inject-fault', {
            target_node_id: nodeId,
            fault_type: faultType,
            magnitude: magnitude,
            duration_ticks: duration
        });
        return response.data;
    },

    runSimulation: async (ticks = 1, applyMitigation = false) => {
        const response = await api.post('/run-simulation', {
            ticks: ticks,
            apply_mitigation: applyMitigation
        });
        return response.data;
    },

    getSystemRisk: async () => {
        const response = await api.get('/system-risk');
        return response.data;
    },

    getEvaluationReport: async () => {
        const response = await api.get('/evaluation-report');
        return response.data;
    },

    runBenchmarkSuite: async (config) => {
        const response = await api.post('/run-benchmark', config);
        return response.data;
    }
};

export default api;
