import React from 'react';
import { AlertTriangle, Activity, Target } from 'lucide-react';
import clsx from 'clsx';

export default function MetricsDashboard({ systemRisk, evaluation, selectedNodeData }) {

    const severityColors = {
        'LOW': 'text-green-500',
        'MODERATE': 'text-yellow-500',
        'HIGH': 'text-orange-500',
        'CRITICAL': 'text-red-500'
    };

    const severityMode = systemRisk?.severity_level || 'LOW';

    return (
        <div className="h-full flex flex-col gap-6">

            <div className="bg-gray-900 p-6 rounded-lg border border-gray-800">
                <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-purple-500" /> System State
                </h3>

                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                        <p className="text-gray-400 text-sm mb-1">Global Severity</p>
                        <p className={clsx("text-2xl font-bold", severityColors[severityMode])}>
                            {severityMode}
                        </p>
                    </div>
                    <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                        <p className="text-gray-400 text-sm mb-1">System Risk Score</p>
                        <p className="text-2xl font-bold text-white">
                            {systemRisk?.system_risk_score ? systemRisk.system_risk_score.toFixed(3) : '0.000'}
                        </p>
                    </div>
                    <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 col-span-2">
                        <p className="text-gray-400 text-sm mb-1">Predicted Failing Nodes</p>
                        <p className="text-2xl font-bold text-white">
                            {systemRisk?.predicted_cascade_size || 0}
                        </p>
                    </div>
                </div>
            </div>

            {selectedNodeData && (
                <div className="bg-gray-900 p-6 rounded-lg border border-gray-800 flex-1">
                    <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
                        <Target className="w-5 h-5 text-blue-500" /> Node Inspector
                    </h3>
                    <div className="space-y-4">
                        <div className="flex justify-between items-center border-b border-gray-800 pb-2">
                            <span className="text-gray-400">Node ID</span>
                            <span className="text-white font-mono">{selectedNodeData.id}</span>
                        </div>
                        <div className="flex justify-between items-center border-b border-gray-800 pb-2">
                            <span className="text-gray-400">Calculated Risk</span>
                            <span className="text-white font-mono">{selectedNodeData.risk.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between items-center border-b border-gray-800 pb-2">
                            <span className="text-gray-400">Criticality Scope</span>
                            <span className="text-white font-mono">{selectedNodeData.criticality_score || 1.0}</span>
                        </div>
                    </div>
                </div>
            )}

            <div className="bg-gray-900 p-6 rounded-lg border border-gray-800 flex-1">
                <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-rose-500" /> Research Evaluation
                </h3>

                <div className="space-y-4">
                    {evaluation ? (
                        <>
                            <div className="flex justify-between items-center">
                                <span className="text-gray-400">Precision</span>
                                <span className="text-white bg-gray-800 px-2 py-1 rounded text-sm">{evaluation.precision.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-gray-400">Recall</span>
                                <span className="text-white bg-gray-800 px-2 py-1 rounded text-sm">{evaluation.recall.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-gray-400">Cascade Deviation</span>
                                <span className="text-white bg-gray-800 px-2 py-1 rounded text-sm">{evaluation.cascade_size_error} nodes</span>
                            </div>
                        </>
                    ) : (
                        <p className="text-gray-500 text-sm">Waiting for prediction stabilization...</p>
                    )}
                </div>
            </div>

        </div>
    );
}
