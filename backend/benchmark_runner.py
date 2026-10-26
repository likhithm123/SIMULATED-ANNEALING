"""
Automated Multi-Seed Scientific Benchmark Harness & Ablation Study Runner
for Adaptive Simulated Annealing (ASA) vs Baselines.

Evaluates algorithms across identical problem instances, initial routes,
and computational budgets across multiple random seeds.
"""

import argparse
import csv
import io
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Scaled canvas Berlin52 optimal closed-loop tour cost (canonical TSPLIB optimum: 7542)
SCALED_BERLIN52_OPTIMAL = 2618.27


def run_headless_simulation(
    cities: List[Dict[str, float]],
    iterations: int,
    algorithm: str = "full",
    initial_route: Optional[List[int]] = None,
    initial_temp: float = 600.0,
    cooling_alpha: float = 0.995,
) -> Dict[str, Any]:
    """
    Executes a single headless optimization run without WebSocket delay.
    Returns complete metrics, decision log, and operator yields.
    """
    from main import (
        route_cost,
        apply_neighborhood_operator,
        get_tour_edges,
        calculate_diversity,
        AdaptiveTemperatureController,
    )

    start_time = time.perf_counter()
    n = len(cities)
    if initial_route is not None:
        route = initial_route.copy()
    else:
        route = list(range(n))
        random.shuffle(route)

    start_cost = route_cost(route, cities)
    current_cost = start_cost
    best_route = route.copy()
    best_cost = start_cost

    evaluations = 0
    accepted_all = 0
    improvements_all = 0
    stagnation = 0
    total_reheats = 0

    WINDOW_SIZE = 50
    recent_accepts = []
    recent_improvements = []
    history_edges: List[set] = [get_tour_edges(route)]

    operator_stats = {
        op: {"proposed": 0, "accepted": 0, "improved": 0}
        for op in ["2-opt", "or-opt", "insert", "swap"]
    }

    # Setup controller or fixed schedule
    is_standard = (algorithm in ("standard_sa", "conventional"))
    ablation_mode = "standard_sa" if is_standard else algorithm

    controller = AdaptiveTemperatureController(
        initial_temperature=initial_temp,
        min_temperature=0.05,
        max_temperature=1200.0,
        ablation_mode=ablation_mode,
    )

    temperature = initial_temp
    op_weights = {"2-opt": 0.65, "or-opt": 0.15, "insert": 0.12, "swap": 0.08}
    time_to_best_iter = 1

    for iteration in range(1, iterations + 1):
        candidate, op_name = apply_neighborhood_operator(route, op_weights)
        evaluations += 1
        operator_stats[op_name]["proposed"] += 1
        candidate_cost = route_cost(candidate, cities)
        delta = candidate_cost - current_cost

        is_accepted = False
        is_improved = False

        curr_t = temperature if is_standard else max(controller.temperature, 1e-6)
        if delta < 0 or (curr_t > 1e-6 and random.random() < math.exp(-delta / curr_t)):
            route = candidate
            current_cost = candidate_cost
            is_accepted = True
            accepted_all += 1
            operator_stats[op_name]["accepted"] += 1

            history_edges.append(get_tour_edges(route))
            if len(history_edges) > 15:
                history_edges.pop(0)

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_route = candidate.copy()
                is_improved = True
                improvements_all += 1
                operator_stats[op_name]["improved"] += 1
                stagnation = 0
                time_to_best_iter = iteration
            else:
                stagnation += 1
        else:
            stagnation += 1

        recent_accepts.append(1 if is_accepted else 0)
        recent_improvements.append(1 if is_improved else 0)
        if len(recent_accepts) > WINDOW_SIZE:
            recent_accepts.pop(0)
            recent_improvements.pop(0)

        # Update temperature
        if is_standard:
            temperature = max(0.01, temperature * cooling_alpha)
        else:
            # Feedback epoch every 40 iterations
            if iteration % 40 == 0:
                win_acc = sum(recent_accepts) / len(recent_accepts)
                win_imp = sum(recent_improvements) / len(recent_improvements)
                diversity = calculate_diversity(history_edges, n)

                decision_data = controller.update(
                    iteration=iteration,
                    window_acceptance_rate=win_acc,
                    window_improvement_rate=win_imp,
                    stagnation=stagnation,
                    diversity=diversity,
                    current_best_cost=best_cost,
                )

                if "REHEAT" in decision_data["decision"] or "DIVERSIFICATION" in decision_data["decision"]:
                    total_reheats += 1
                    if decision_data["decision"] == "STRUCTURAL_DIVERSIFICATION":
                        op_weights = {"2-opt": 0.35, "or-opt": 0.30, "insert": 0.20, "swap": 0.15}
                    else:
                        op_weights = {"2-opt": 0.50, "or-opt": 0.22, "insert": 0.18, "swap": 0.10}
                elif decision_data["mode"] == "EXPLOITATION":
                    op_weights = {"2-opt": 0.80, "or-opt": 0.12, "insert": 0.05, "swap": 0.03}
                else:
                    op_weights = {"2-opt": 0.65, "or-opt": 0.15, "insert": 0.12, "swap": 0.08}

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    improvement_pct = round(((start_cost - best_cost) / start_cost) * 100, 2) if start_cost > 0 else 0.0

    return {
        "algorithm": algorithm,
        "initial_cost": round(start_cost, 2),
        "best_cost": round(best_cost, 2),
        "improvement_pct": improvement_pct,
        "runtime_ms": elapsed_ms,
        "evaluations": evaluations,
        "total_reheats": total_reheats,
        "time_to_best_iter": time_to_best_iter,
        "final_route": best_route,
        "decision_log": controller.decision_history if not is_standard else [],
        "operator_stats": operator_stats,
    }


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Computes mean, standard deviation, median, IQR, min, and max."""
    arr = np.array(values, dtype=float)
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "mean": round(float(np.mean(arr)), 2),
        "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 2),
        "median": round(float(np.median(arr)), 2),
        "iqr": round(float(q75 - q25), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
    }


def run_multi_seed_benchmark(
    distribution: str = "berlin52",
    city_count: int = 52,
    iterations: int = 1500,
    seeds: Optional[List[int]] = None,
    algorithms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Executes a multi-seed scientific benchmark suite across requested algorithms
    using strictly identical initial tours and coordinates per seed.
    """
    from main import generate_cities

    if seeds is None:
        seeds = [42, 101, 777, 999, 1234]
    if algorithms is None:
        algorithms = [
            "standard_sa",
            "acceptance_only",
            "full",
            "no_diversity",
            "no_stagnation",
            "no_improvement",
        ]

    is_berlin = (distribution == "berlin52")
    actual_city_count = 52 if is_berlin else city_count

    raw_runs: List[Dict[str, Any]] = []
    algo_results: Dict[str, List[Dict[str, Any]]] = {algo: [] for algo in algorithms}

    for seed in seeds:
        cities = generate_cities(actual_city_count, seed=seed, distribution=distribution)
        n = len(cities)
        # Fix identical initial permutation for this seed across all tested algorithms
        rng = random.Random(seed)
        shared_initial_route = list(range(n))
        rng.shuffle(shared_initial_route)

        for algo in algorithms:
            # Set deterministic random seed for algorithm move proposals
            random.seed(seed * 31 + hash(algo) % 10000)
            res = run_headless_simulation(
                cities=cities,
                iterations=iterations,
                algorithm=algo,
                initial_route=shared_initial_route,
            )
            res["seed"] = seed
            res["distribution"] = distribution
            if is_berlin:
                res["optimality_gap_pct"] = round(
                    ((res["best_cost"] - SCALED_BERLIN52_OPTIMAL) / SCALED_BERLIN52_OPTIMAL) * 100, 2
                )
            else:
                res["optimality_gap_pct"] = None

            raw_runs.append(res)
            algo_results[algo].append(res)

    # Compute statistical aggregation per algorithm
    summary: Dict[str, Any] = {}
    for algo, runs in algo_results.items():
        costs = [r["best_cost"] for r in runs]
        runtimes = [r["runtime_ms"] for r in runs]
        reheats = [r["total_reheats"] for r in runs]
        ttb = [r["time_to_best_iter"] for r in runs]
        gaps = [r["optimality_gap_pct"] for r in runs if r["optimality_gap_pct"] is not None]

        cost_stats = compute_statistics(costs)
        summary[algo] = {
            "algorithm": algo,
            "cost_stats": cost_stats,
            "mean_cost": cost_stats["mean"],
            "std_cost": cost_stats["std"],
            "median_cost": cost_stats["median"],
            "iqr_cost": cost_stats["iqr"],
            "min_cost": cost_stats["min"],
            "max_cost": cost_stats["max"],
            "mean_runtime_ms": round(float(np.mean(runtimes)), 2),
            "mean_reheats": round(float(np.mean(reheats)), 2),
            "mean_time_to_best": round(float(np.mean(ttb)), 1),
            "mean_optimality_gap_pct": round(float(np.mean(gaps)), 2) if gaps else None,
            "runs_count": len(runs),
        }

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "distribution": distribution,
        "city_count": actual_city_count,
        "iterations": iterations,
        "seeds": seeds,
        "algorithms": algorithms,
        "summary": summary,
        "raw_runs": raw_runs,
    }


def export_benchmark_to_csv(benchmark_data: Dict[str, Any]) -> str:
    """Formats benchmark results as a CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Distribution",
        "CityCount",
        "Iterations",
        "Seed",
        "Algorithm",
        "InitialCost",
        "BestCost",
        "ImprovementPct",
        "OptimalityGapPct",
        "RuntimeMs",
        "TotalReheats",
        "TimeToBestIter",
        "Evaluations",
    ])

    for r in benchmark_data.get("raw_runs", []):
        writer.writerow([
            r.get("distribution", ""),
            benchmark_data.get("city_count", 0),
            benchmark_data.get("iterations", 0),
            r.get("seed", ""),
            r.get("algorithm", ""),
            r.get("initial_cost", ""),
            r.get("best_cost", ""),
            r.get("improvement_pct", ""),
            r.get("optimality_gap_pct", ""),
            r.get("runtime_ms", ""),
            r.get("total_reheats", ""),
            r.get("time_to_best_iter", ""),
            r.get("evaluations", ""),
        ])

    return output.getvalue()


def print_benchmark_table(benchmark_data: Dict[str, Any]):
    """Pretty prints the benchmark comparison table to stdout."""
    dist = benchmark_data["distribution"]
    cities = benchmark_data["city_count"]
    iters = benchmark_data["iterations"]
    seeds = benchmark_data["seeds"]

    print("\n" + "=" * 80)
    print(f" EXPERIMENTAL BENCHMARK & ABLATION STUDY RESULTS ")
    print(f" Topology: {dist} ({cities} cities) | Budget: {iters} iterations | Seeds: {len(seeds)}")
    print("=" * 80)

    header = f"{'Algorithm':<22} | {'Mean Cost ± Std':<20} | {'Median (IQR)':<18} | {'Gap %':<8} | {'Reheats':<7} | {'Time (ms)':<9}"
    print(header)
    print("-" * 80)

    summary = benchmark_data["summary"]
    # Sort by mean cost
    sorted_algos = sorted(summary.values(), key=lambda x: x["mean_cost"])
    for s in sorted_algos:
        algo_name = s["algorithm"]
        cost_str = f"{s['mean_cost']} ± {s['std_cost']}"
        med_str = f"{s['median_cost']} ({s['iqr_cost']})"
        gap_str = f"{s['mean_optimality_gap_pct']}%" if s["mean_optimality_gap_pct"] is not None else "N/A"
        reheat_str = f"{s['mean_reheats']}"
        time_str = f"{s['mean_runtime_ms']} ms"
        print(f"{algo_name:<22} | {cost_str:<20} | {med_str:<18} | {gap_str:<8} | {reheat_str:<7} | {time_str:<9}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Seed Scientific Benchmark Runner for ASA")
    parser.add_argument("--distribution", type=str, default="berlin52", help="Topology (berlin52, clustered, random, grid, circular)")
    parser.add_argument("--cities", type=int, default=52, help="Number of cities")
    parser.add_argument("--iterations", type=int, default=1500, help="Evaluation iterations")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV output file path")

    args = parser.parse_args()
    seed_list = [42 + i * 137 for i in range(args.seeds)]

    print(f"Executing scientific benchmark suite across {args.seeds} seeds...")
    results = run_multi_seed_benchmark(
        distribution=args.distribution,
        city_count=args.cities,
        iterations=args.iterations,
        seeds=seed_list,
    )

    print_benchmark_table(results)

    if args.output:
        csv_content = export_benchmark_to_csv(results)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(csv_content)
        print(f"Results exported to {args.output}")
