import React, { useState } from 'react';
import { simulationApi } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { Play, TrendingUp, ShieldAlert, BarChart2 } from 'lucide-react';
import clsx from 'clsx';

export default function BenchmarkPage() {
    const [isRunning, setIsRunning] = useState(false);
    const [results, setResults] = useState(null);
    const [config, setConfig] = useState({
        num_services: 12,
        density: 0.4,
        fault_type: "cpu_spike",
        fault_duration: 4,
        learning_cycles: 10,
        repeated_trials: 2
    });

    const handleRunBenchmark = async () => {
        setIsRunning(true);
        try {
            const data = await simulationApi.runBenchmarkSuite(config);
            setResults(data);
        } catch (e) {
            console.error("Benchmark failed:", e);
        } finally {
            setIsRunning(false);
        }
    };

    const renderCascadeComparisonChart = () => {
        if (!results) return null;
        const data = [
            {
                name: "Unmitigated Baseline",
                "Cascade Size": results.no_mitigation_metrics.average_cascade_size
            },
            {
                name: "Static Action Engine",
                "Cascade Size": results.static_mitigation_metrics.average_cascade_size
            },
            {
                name: "Adaptive Learning Loop",
                "Cascade Size": results.adaptive_learning_metrics.average_cascade_size
            }
        ];

        return (
            <div className="bg-gray-900 border border-gray-800 p-6 rounded-lg h-96">
                <h3 className="text-white font-medium mb-4 flex items-center gap-2">
                    <BarChart2 className="text-blue-500 w-5 h-5" /> Mode comparison: Average Failing Nodes
                </h3>
                <ResponsiveContainer width="100%" height="85%">
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="name" stroke="#9ca3af" tick={{ fill: '#9ca3af' }} />
                        <YAxis stroke="#9ca3af" tick={{ fill: '#9ca3af' }} />
                        <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }} />
                        <Bar dataKey="Cascade Size" fill="#3b82f6" radius={[4, 4, 0, 0]} barSize={60} />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        );
    };

    const renderPrecisionTrendChart = () => {
        if (!results) return null;
        const trend = results.adaptive_learning_metrics.convergence_trend;
        const data = trend.map((val, idx) => ({
            cycle: `Cycle ${idx + 1}`,
            Precision: val
        }));

        return (
            <div className="bg-gray-900 border border-gray-800 p-6 rounded-lg h-96">
                <h3 className="text-white font-medium mb-4 flex items-center gap-2">
                    <TrendingUp className="text-purple-500 w-5 h-5" /> Precision Convergence over Time
                </h3>
                <ResponsiveContainer width="100%" height="85%">
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="cycle" stroke="#9ca3af" tick={{ fill: '#9ca3af' }} />
                        <YAxis stroke="#9ca3af" tick={{ fill: '#9ca3af' }} domain={[0, 1]} />
                        <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }} />
                        <Line type="monotone" dataKey="Precision" stroke="#a855f7" strokeWidth={3} dot={{ r: 4, fill: '#a855f7' }} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        );
    };

    const renderSeverityDistribution = () => {
        if (!results) return null;
        // Compare No Mitigation vs Adaptive Learning Distributions
        const noSev = results.no_mitigation_metrics.severity_distribution;
        const adSev = results.adaptive_learning_metrics.severity_distribution;

        const COLORS = {
            'LOW': '#10b981',
            'MODERATE': '#eab308',
            'HIGH': '#f97316',
            'CRITICAL': '#ef4444'
        };

        const renderPie = (distDict, title) => {
            const data = Object.keys(distDict).map(k => ({ name: k, value: distDict[k] })).filter(d => d.value > 0);
            return (
                <div className="flex-1 text-center">
                    <p className="text-gray-400 text-sm mb-2">{title}</p>
                    <ResponsiveContainer width="100%" height={240}>
                        <PieChart>
                            <Pie data={data} cx="50%" cy="50%" innerRadius={60} outerRadius={80} dataKey="value" stroke="none">
                                {data.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[entry.name]} />)}
                            </Pie>
                            <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }} />
                            <Legend />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
            );
        };

        return (
            <div className="bg-gray-900 border border-gray-800 p-6 rounded-lg">
                <h3 className="text-white font-medium mb-4 flex items-center gap-2">
                    <ShieldAlert className="text-rose-500 w-5 h-5" /> Severity Output Distribution
                </h3>
                <div className="flex justify-around items-center h-72">
                    {renderPie(noSev, "No Mitigation")}
                    {renderPie(adSev, "Adaptive Learning")}
                </div>
            </div>
        );
    };

    return (
        <div className="p-8 h-full overflow-y-auto custom-scrollbar flex flex-col gap-6 w-full max-w-7xl mx-auto">

            <div className="flex items-center justify-between bg-gray-900 p-6 rounded-lg border border-gray-800">
                <div>
                    <h2 className="text-2xl font-bold text-white tracking-tight">Experiment Benchmark Runner</h2>
                    <p className="text-gray-400 mt-1">Batch executes synthetic topological failures across mitigation strategies.</p>
                </div>

                <div className="flex gap-4 items-center">
                    <div className="flex flex-col">
                        <label className="text-xs text-gray-500">Learning Cycles</label>
                        <input type="number" value={config.learning_cycles} onChange={e => setConfig({ ...config, learning_cycles: parseInt(e.target.value) })} className="bg-gray-800 text-white px-3 py-1.5 rounded border border-gray-700 w-24 text-sm focus:outline-none focus:border-blue-500" />
                    </div>
                    <button
                        onClick={handleRunBenchmark}
                        disabled={isRunning}
                        className={clsx(
                            "py-2 px-6 rounded font-medium flex items-center gap-2 transition-colors h-10 mt-4",
                            isRunning ? "bg-gray-700 text-gray-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-500 text-white"
                        )}
                    >
                        {isRunning ? 'Running...' : <><Play className="w-4 h-4" /> Run Suite</>}
                    </button>
                </div>
            </div>

            {!results && !isRunning && (
                <div className="flex flex-1 items-center justify-center text-gray-500">
                    <p>Trigger "Run Suite" to generate benchmark analytics.</p>
                </div>
            )}

            {isRunning && (
                <div className="flex flex-1 items-center justify-center">
                    <div className="flex flex-col items-center gap-4">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
                        <p className="text-gray-400 animate-pulse">Computing simulation cascades across {config.repeated_trials * config.learning_cycles} scenarios...</p>
                    </div>
                </div>
            )}

            {results && !isRunning && (
                <div className="flex flex-col gap-6 pb-12">
                    <div className="grid grid-cols-3 gap-6">
                        <div className="bg-blue-900/30 border border-blue-800/50 p-6 rounded-lg col-span-1">
                            <p className="text-blue-300 text-sm font-medium mb-1">Total Synthetic Scenarios</p>
                            <p className="text-4xl font-bold text-white">{results.total_scenarios_per_track}</p>
                        </div>
                        <div className="bg-emerald-900/30 border border-emerald-800/50 p-6 rounded-lg col-span-1">
                            <p className="text-emerald-300 text-sm font-medium mb-1">Adaptive Risk Reduction</p>
                            <p className="text-4xl font-bold text-emerald-400">{results.adaptive_learning_metrics.average_cascade_reduction_percent}%</p>
                        </div>
                        <div className="bg-purple-900/30 border border-purple-800/50 p-6 rounded-lg col-span-1">
                            <p className="text-purple-300 text-sm font-medium mb-1">Adaptive Peak Precision</p>
                            <p className="text-4xl font-bold text-purple-400">
                                {results.adaptive_learning_metrics.convergence_trend[results.adaptive_learning_metrics.convergence_trend.length - 1]}
                            </p>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-6">
                        {renderCascadeComparisonChart()}
                        {renderPrecisionTrendChart()}
                    </div>

                    {renderSeverityDistribution()}
                </div>
            )}

        </div>
    );
}
