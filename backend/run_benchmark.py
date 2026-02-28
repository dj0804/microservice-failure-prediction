import json
from simulation.benchmark_runner import BenchmarkRunner

def main():
    print("========================================")
    print("🧪 Failure Propagation Benchmark Runner")
    print("========================================\n")
    
    config = {
        "num_services": 12,
        "density": 0.4,
        "fault_type": "cpu_spike",
        "fault_duration": 4,
        "learning_cycles": 15,
        "repeated_trials": 2
    }
    
    print(f"Executing {config['repeated_trials'] * config['learning_cycles']} Synthetic Scenarios...")
    runner = BenchmarkRunner(config)
    results = runner.run_benchmark()
    
    print("\n📊 NO MITIGATION (BASELINE)")
    base = results["no_mitigation_metrics"]
    print(f"Average Cascade Size: {base['average_cascade_size']} nodes")
    print(f"Average Precision: {base['average_precision']}")
    print(f"Severity Distribution: {base['severity_distribution']}\n")
    
    print("📊 WITH PREVENTIVE ACTIONS (STATIC)")
    stat = results["static_mitigation_metrics"]
    print(f"Average Cascade Size: {stat['average_cascade_size']} nodes")
    print(f"Average Risk Reduction: {stat['average_cascade_reduction_percent']}%")
    print(f"Severity Distribution: {stat['severity_distribution']}\n")
    
    print("📊 WITH ADAPTIVE FEEDBACK LEARNING")
    adap = results["adaptive_learning_metrics"]
    print(f"Average Cascade Size: {adap['average_cascade_size']} nodes")
    print(f"Average Risk Reduction: {adap['average_cascade_reduction_percent']}%")
    print(f"Average Precision: {adap['average_precision']}")
    print(f"Precision Convergence Trend (over {config['learning_cycles']} cycles):")
    print(" → ".join(map(str, adap['convergence_trend'])))
    print(f"Severity Distribution: {adap['severity_distribution']}\n")

if __name__ == "__main__":
    main()
