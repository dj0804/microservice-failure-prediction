import React, { useState, useEffect } from 'react';
import GraphView from '../components/GraphView';
import SimulationControls from '../components/SimulationControls';
import MetricsDashboard from '../components/MetricsDashboard';
import { simulationApi } from '../services/api';

export default function SimulationPage() {
    const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
    const [systemRisk, setSystemRisk] = useState(null);
    const [evaluation, setEvaluation] = useState(null);
    const [selectedNodeId, setSelectedNodeId] = useState(null);

    const [isRunning, setIsRunning] = useState(false);
    const [applyMitigation, setApplyMitigation] = useState(false);

    // Auto-fetch graph state whenever we run actions
    const refreshState = async () => {
        try {
            const graph = await simulationApi.getGraphState();
            setGraphData(graph);

            const risk = await simulationApi.getSystemRisk();
            setSystemRisk(risk);

            try {
                const evalReport = await simulationApi.getEvaluationReport();
                // Eval report throws 404 or missing if no prediction snapshot is available yet
                if (!evalReport.error) {
                    setEvaluation(evalReport.evaluation);
                }
            } catch (e) {
                // Ignore eval errors if simulation hasn't predicted yet
                setEvaluation(null);
            }
        } catch (error) {
            console.error("Failed to fetch state:", error);
        }
    };

    // On mount, check if backend already has a graph
    useEffect(() => {
        refreshState();
    }, []);

    const handleInitGraph = async (numNodes) => {
        setIsRunning(true);
        try {
            await simulationApi.initializeGraph(numNodes, 0.3);
            await refreshState();
        } finally {
            setIsRunning(false);
        }
    };

    const handleInjectFault = async (nodeId, faultType) => {
        setIsRunning(true);
        try {
            // High magnitude for visual impact guaranteed
            await simulationApi.injectFault(nodeId, faultType, 1.0, 5);
            await refreshState();
        } finally {
            setIsRunning(false);
        }
    };

    const handleRunTick = async () => {
        setIsRunning(true);
        try {
            await simulationApi.runSimulation(1, applyMitigation);
            await refreshState();
        } finally {
            setIsRunning(false);
        }
    };

    const selectedNodeData = graphData?.nodes?.find(n => n.id === selectedNodeId);

    return (
        <div className="flex h-full p-6 gap-6 overflow-hidden">
            {/* Left Sidebar - Controls and Metrics */}
            <div className="w-[450px] flex flex-col gap-6 overflow-y-auto pr-2 pb-10 custom-scrollbar">
                <SimulationControls
                    onInit={handleInitGraph}
                    onFault={handleInjectFault}
                    onRun={handleRunTick}
                    applyMitigation={applyMitigation}
                    setApplyMitigation={setApplyMitigation}
                    isRunning={isRunning}
                />

                <MetricsDashboard
                    systemRisk={systemRisk}
                    evaluation={evaluation}
                    selectedNodeData={selectedNodeData ? { id: selectedNodeData.id, risk: selectedNodeData.calculated_risk_score } : null}
                />
            </div>

            {/* Right Main Area - Graph Visualization */}
            <div className="flex-1 rounded-lg overflow-hidden border border-gray-800 shadow-xl">
                <GraphView
                    graphData={graphData}
                    selectedNode={selectedNodeId}
                    onNodeClick={setSelectedNodeId}
                />
            </div>
        </div>
    );
}
