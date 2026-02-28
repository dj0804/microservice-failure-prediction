import React, { useState } from 'react';
import { Play, Settings, Zap, Shield, RotateCcw } from 'lucide-react';

export default function SimulationControls({
    onInit,
    onFault,
    onRun,
    applyMitigation,
    setApplyMitigation,
    isRunning
}) {
    const [numNodes, setNumNodes] = useState(12);
    const [faultNode, setFaultNode] = useState('node_0');
    const [faultType, setFaultType] = useState('cpu_spike');

    return (
        <div className="bg-gray-900 p-6 rounded-lg border border-gray-800 flex flex-col gap-6">
            <div>
                <h3 className="text-lg font-medium text-white flex items-center gap-2 mb-4">
                    <Settings className="w-5 h-5" /> Environment Setup
                </h3>
                <div className="flex flex-col gap-3">
                    <div className="flex items-center justify-between">
                        <label className="text-sm text-gray-400">Node Count:</label>
                        <input
                            type="number"
                            className="bg-gray-800 text-white px-3 py-1.5 rounded border border-gray-700 w-24 focus:outline-none focus:border-blue-500"
                            value={numNodes}
                            onChange={(e) => setNumNodes(parseInt(e.target.value))}
                        />
                    </div>
                    <button
                        onClick={() => onInit(numNodes)}
                        disabled={isRunning}
                        className="w-full bg-gray-800 hover:bg-gray-700 text-white font-medium py-2 px-4 rounded transition-colors disabled:opacity-50"
                    >
                        Initialize Graph
                    </button>
                </div>
            </div>

            <div className="h-px bg-gray-800 w-full" />

            <div>
                <h3 className="text-lg font-medium text-white flex items-center gap-2 mb-4">
                    <Zap className="w-5 h-5 text-yellow-500" /> Fault Injection
                </h3>
                <div className="flex flex-col gap-3">
                    <div className="flex items-center justify-between">
                        <label className="text-sm text-gray-400">Target Node:</label>
                        <input
                            type="text"
                            className="bg-gray-800 text-white px-3 py-1.5 rounded border border-gray-700 w-24 focus:outline-none focus:border-blue-500"
                            value={faultNode}
                            onChange={(e) => setFaultNode(e.target.value)}
                        />
                    </div>
                    <div className="flex items-center justify-between">
                        <label className="text-sm text-gray-400">Fault Type:</label>
                        <select
                            className="bg-gray-800 text-white px-3 py-1.5 rounded border border-gray-700 w-32 focus:outline-none focus:border-blue-500"
                            value={faultType}
                            onChange={(e) => setFaultType(e.target.value)}
                        >
                            <option value="cpu_spike">CPU Spike</option>
                            <option value="latency_spike">Latency</option>
                            <option value="error_spike">Error Rate</option>
                        </select>
                    </div>
                    <button
                        onClick={() => onFault(faultNode, faultType)}
                        disabled={isRunning}
                        className="w-full bg-yellow-600 hover:bg-yellow-500 text-white font-medium py-2 px-4 rounded transition-colors disabled:opacity-50"
                    >
                        Inject Fault
                    </button>
                </div>
            </div>

            <div className="h-px bg-gray-800 w-full" />

            <div>
                <h3 className="text-lg font-medium text-white flex items-center gap-2 mb-4">
                    <Play className="w-5 h-5 text-green-500" /> Execution
                </h3>

                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                        <Shield className="w-4 h-4 text-blue-400" />
                        <label className="text-sm text-gray-300 cursor-pointer select-none">Action Engine</label>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input
                            type="checkbox"
                            className="sr-only peer"
                            checked={applyMitigation}
                            onChange={(e) => setApplyMitigation(e.target.checked)}
                        />
                        <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                    </label>
                </div>

                <button
                    onClick={onRun}
                    disabled={isRunning}
                    className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-2 px-4 rounded flex justify-center items-center gap-2 transition-colors disabled:opacity-50"
                >
                    {isRunning ? <RotateCcw className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />}
                    Step Next Tick
                </button>
            </div>
        </div>
    );
}
