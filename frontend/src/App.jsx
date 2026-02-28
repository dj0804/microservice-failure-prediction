import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import SimulationPage from './pages/SimulationPage';
import BenchmarkPage from './pages/BenchmarkPage';
import { Activity } from 'lucide-react';

function App() {
    return (
        <Router>
            <div className="flex flex-col h-screen bg-gray-950 text-gray-100 overflow-hidden font-sans">
                <header className="flex items-center justify-between px-6 py-4 bg-gray-900 border-b border-gray-800">
                    <div className="flex items-center gap-2">
                        <Activity className="text-blue-500" />
                        <h1 className="text-xl font-semibold tracking-tight">Proactive Failure Management</h1>
                    </div>
                    <nav className="flex gap-4">
                        <Link to="/" className="px-3 py-1.5 text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-800 rounded-md transition-colors">Simulation</Link>
                        <Link to="/benchmark" className="px-3 py-1.5 text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-800 rounded-md transition-colors">Benchmark</Link>
                    </nav>
                </header>
                <main className="flex-1 overflow-hidden">
                    <Routes>
                        <Route path="/" element={<SimulationPage />} />
                        <Route path="/benchmark" element={<BenchmarkPage />} />
                    </Routes>
                </main>
            </div>
        </Router>
    );
}

export default App;
