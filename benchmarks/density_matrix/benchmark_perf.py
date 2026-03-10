"""
Performance Benchmark: SQUANDER vs Qiskit Noisy Circuit Simulation

Benchmarks execution time of identical circuits in both frameworks.

Run with: python benchmarks/benchmark_perf.py
"""

import time

import numpy as np
from circuits import BENCHMARK_CIRCUITS, CIRCUITS_BY_QUBITS

from squander.density_matrix import DensityMatrix

# Benchmark configuration
NUM_RUNS = 1  # Number of timed runs per circuit
WARMUP_RUNS = 0  # Number of warmup runs before timing


def benchmark_func(func, num_runs=10, warmup=3):
    """Benchmark a function with warmup runs."""
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def run_benchmark(circuits, num_runs=10, warmup=3):
    """Run benchmark on a list of circuits."""
    results = []

    for name, builder_fn in circuits:
        # Build circuits once
        builder = builder_fn()

        # Benchmark SQUANDER
        def run_sq(b=builder):
            rho = DensityMatrix(b.n)
            b.sq.apply_to(np.array([]), rho)

        sq_time = benchmark_func(run_sq, num_runs=num_runs, warmup=warmup)

        # Benchmark Qiskit using the same method as validation (AerSimulator)
        # Note: QiskitDensityMatrix.evolve() cannot handle QuantumError noise channels
        def run_qk(b=builder):
            b.run_qiskit()

        qk_time = benchmark_func(run_qk, num_runs=num_runs, warmup=warmup)

        speedup = qk_time["mean"] / sq_time["mean"]
        results.append(
            {
                "name": name,
                "qubits": builder.n,
                "ops": len(builder.ops),
                "squander_ms": sq_time["mean"],
                "squander_std": sq_time["std"],
                "qiskit_ms": qk_time["mean"],
                "qiskit_std": qk_time["std"],
                "speedup": speedup,
            }
        )

    return results


def print_results(results, title="Benchmark Results"):
    """Print benchmark results in a formatted table."""
    print(
        f"\n{'Circuit':<20} {'Qubits':<8} {'Ops':<6} {'SQUANDER(ms)':<14} {'Qiskit(ms)':<14} {'Speedup':<10}"
    )
    print("-" * 75)

    for r in results:
        print(
            f"  {r['name']:<18} {r['qubits']:<8} {r['ops']:<6} "
            f"{r['squander_ms']:>10.3f}    {r['qiskit_ms']:>10.3f}    {r['speedup']:>6.1f}x"
        )


def print_summary(results):
    """Print benchmark summary statistics."""
    speedups = [r["speedup"] for r in results]

    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)

    print(f"\n  Average speedup: {np.mean(speedups):.1f}x faster")

    best = max(results, key=lambda x: x["speedup"])
    worst = min(results, key=lambda x: x["speedup"])

    print(f"  Best speedup:    {best['speedup']:.1f}x ({best['name']})")
    print(f"  Worst speedup:   {worst['speedup']:.1f}x ({worst['name']})")


def main():
    print("=" * 70)
    print("  PERFORMANCE BENCHMARK: SQUANDER vs Qiskit")
    print("=" * 70)
    print(f"\nMeasuring execution time ({NUM_RUNS} run(s), {WARMUP_RUNS} warmup)...")

    # Run benchmark on representative circuits
    results = run_benchmark(BENCHMARK_CIRCUITS, num_runs=NUM_RUNS, warmup=WARMUP_RUNS)

    # Print results
    print_results(results)
    print_summary(results)

    print("\n" + "=" * 70)

    # Optional: Run full benchmark by qubit count
    print("\n\nDetailed benchmark by qubit count:")

    for n_qubits in sorted(CIRCUITS_BY_QUBITS.keys()):
        circuits = CIRCUITS_BY_QUBITS[n_qubits]
        print(f"\n--- {n_qubits}-QUBIT CIRCUITS ---")
        results = run_benchmark(circuits, num_runs=NUM_RUNS, warmup=WARMUP_RUNS)

        for r in results:
            print(
                f"  {r['name']:<20} SQUANDER: {r['squander_ms']:>8.3f}ms  "
                f"Qiskit: {r['qiskit_ms']:>10.3f}ms  Speedup: {r['speedup']:>8.1f}x"
            )

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
