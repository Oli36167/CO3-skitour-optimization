"""
This code evaluates
- how well SA performs compared to Dijkstra: cost ratio
- how stable that performance is: sd
- on three routes

1. Fixed-parameter robustness
- Dijkstra as optimal cost
- run SA 10 times (one per seed) with default parameters (T0=4000, alpha=0.998,
    2000 iterations)
        - cost ratio (SA/Dijkstra)
        - mean, sd, coefficient of variation for each route
        - comparing across routes
        - boxplots
        - convergence curve (improvement of SA's cost function over iterations
            within a single run)

2. Parameter sensitivity
- for each route, it sweeps one parameter at a time
- T0 over {2000, 4000, 8000}, alpha over {0.995, 0.998, 0.999},
   and iterations over {1000, 2000, 4000}
- holding the other two at their defaults
- each configuration runs 10 times (one per seed)
- boxplots per parameter

Outputs:
- A cross-route boxplot
- convergence curves
- per-route parameter boxplots
- a console report with verdicts
- and a CSV table with all the sweep results
"""

import csv
import math
import random
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from dijkstra_terrain_graph import dijkstra
from joblib import Parallel, delayed
from load_elevation_data import load_elevation_data
from simulated_annealing_a_to_b import path_cost, simulated_annealing
from terrain_graph import TerrainGraph

## -----------------------------------------------------------
# Defaults (held fixed while sweeping the other parameter)
# -----------------------------------------------------------
DEFAULT_T0 = 4000
DEFAULT_ALPHA = 0.998
DEFAULT_ITERS = 2000


def run_single_trial(terrain, start, goal, T0, alpha, iterations, seed):
    random.seed(seed)
    t0 = time.time()
    try:
        _, cost, initial_path, _ = simulated_annealing(
            terrain, start, goal, T0=T0, alpha=alpha, iterations=iterations
        )
        elapsed = time.time() - t0
        init_cost = path_cost(terrain, initial_path)
        return {
            "cost": cost,
            "init_cost": init_cost,
            "improvement": 1 - cost / init_cost if init_cost > 0 else 0,
            "time_s": elapsed,
            "valid": math.isfinite(cost),
        }
    except Exception as e:
        print(f"  Trial failed (seed={seed}): {e}")
        return {
            "cost": float("inf"),
            "init_cost": float("inf"),
            "improvement": 0,
            "time_s": time.time() - t0,
            "valid": False,
        }


def compute_stats(results, opt_cost):
    valid = [r for r in results if r["valid"]]
    costs = [r["cost"] for r in valid]
    if not costs:
        return None
    avg = sum(costs) / len(costs)
    best = min(costs)
    worst = max(costs)
    std = (sum((c - avg) ** 2 for c in costs) / len(costs)) ** 0.5
    ratio = avg / opt_cost if opt_cost > 0 else float("inf")
    avg_time = sum(r["time_s"] for r in valid) / len(valid)
    avg_improve = sum(r["improvement"] for r in valid) / len(valid)
    return {
        "avg": avg,
        "best": best,
        "worst": worst,
        "std": std,
        "ratio": ratio,
        "avg_improve": avg_improve,
        "avg_time": avg_time,
        "valid": len(valid),
        "total": len(results),
    }


def sweep_one_param(terrain, start, goal, opt_cost, param_name, values, seeds):
    sweep_results = []
    raw_results = []  # list of (param_name, param_value, seed, ratio)
    for val in values:
        T0 = val if param_name == "T0" else DEFAULT_T0
        alpha = val if param_name == "alpha" else DEFAULT_ALPHA
        iters = val if param_name == "iterations" else DEFAULT_ITERS
        results = Parallel(n_jobs=-1)(
            delayed(run_single_trial)(
                terrain, start, goal, T0, alpha, iters, seed
            )
            for seed in seeds
        )
        for seed, r in zip(seeds, results):
            ratio = (
                r["cost"] / opt_cost
                if r["valid"] and opt_cost > 0
                else float("inf")
            )
            raw_results.append(
                {
                    "param_name": param_name,
                    "param_value": val,
                    "seed": seed,
                    "ratio": ratio,
                    "valid": r["valid"],
                }
            )
        stats = compute_stats(results, opt_cost)
        if stats:
            stats["param_name"] = param_name
            stats["param_value"] = val
            sweep_results.append(stats)
    return sweep_results, raw_results


def convergence_curve(
    terrain, start, goal, opt_cost, seed, max_iters=4000, step=200
):
    random.seed(seed)
    try:
        _, _, _, history = simulated_annealing(
            terrain,
            start,
            goal,
            T0=DEFAULT_T0,
            alpha=DEFAULT_ALPHA,
            iterations=max_iters,
            snapshot_interval=step,
        )
        return [
            {"iterations": h["iteration"], "ratio": h["cost"] / opt_cost}
            for h in history
        ]
    except Exception as e:
        print(f"  Convergence curve failed (seed={seed}): {e}")
        return []


def plot_convergence(route_curves, filename="convergence.png"):
    """Plot convergence curves for all routes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for route_name, curves in route_curves.items():
        valid_curves = [c for c in curves if c]
        if not valid_curves:
            continue
        color = f"C{list(route_curves.keys()).index(route_name)}"
        for curve in valid_curves:
            iters, ratios = zip(*[(p["iterations"], p["ratio"]) for p in curve])
            ax.plot(iters, ratios, alpha=0.4, color=color)
        # plot average
        all_iters = [p["iterations"] for p in valid_curves[0]]
        avg_ratios = []
        for step_idx in range(len(all_iters)):
            vals = [
                c[step_idx]["ratio"]
                for c in valid_curves
                if step_idx < len(c) and c[step_idx]["ratio"] < float("inf")
            ]
            avg_ratios.append(sum(vals) / len(vals) if vals else float("inf"))
        ax.plot(
            all_iters[: len(avg_ratios)],
            avg_ratios,
            linewidth=2.5,
            label=route_name,
            color=color,
        )

    ax.axhline(
        y=1.0,
        color="black",
        linestyle="--",
        alpha=0.5,
        label="Dijkstra optimal",
    )
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost / Dijkstra Optimal")
    ax.set_title("SA Convergence by Route")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"  Convergence plot saved to {filename}")


def plot_parameter_boxplots(all_raw, route_name, filename="boxplots.png"):
    """Boxplot of cost ratios per parameter value — shows distribution, not just mean."""
    grouped = defaultdict(list)
    for r in all_raw:
        grouped[r["param_name"]].append(r)

    param_names = list(grouped.keys())
    fig, axes = plt.subplots(
        1, len(param_names), figsize=(5 * len(param_names), 5)
    )
    if len(param_names) == 1:
        axes = [axes]

    for ax, param_name in zip(axes, param_names):
        entries = grouped[param_name]
        values = sorted(set(e["param_value"] for e in entries))
        data = []
        for val in values:
            ratios = [
                e["ratio"]
                for e in entries
                if e["param_value"] == val and e["ratio"] < float("inf")
            ]
            data.append(ratios)

        bp = ax.boxplot(
            data, patch_artist=True, tick_labels=[str(v) for v in values]
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.7)
        ax.axhline(
            y=1.0,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Dijkstra optimal",
        )
        ax.set_xlabel(param_name)
        ax.set_ylabel("Cost / Dijkstra Optimal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Parameter Robustness: {route_name}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"  Boxplot saved to {filename}")


def save_results_table(all_route_stats, filename="robustness_results.csv"):
    """Save all sweep results across all routes to a single CSV table."""
    fieldnames = [
        "Route",
        "Parameter",
        "Value",
        "Avg Cost",
        "Best Cost",
        "Worst Cost",
        "Std",
        "Ratio to Optimal",
        "Avg Improvement",
        "Avg Time (s)",
        "Valid Runs",
        "Total Runs",
        "Dijkstra Optimal",
    ]
    rows = []
    for route_name, (opt_cost, stats) in all_route_stats.items():
        for s in stats:
            rows.append(
                {
                    "Route": route_name,
                    "Parameter": s["param_name"],
                    "Value": s["param_value"],
                    "Avg Cost": f"{s['avg']:.1f}",
                    "Best Cost": f"{s['best']:.1f}",
                    "Worst Cost": f"{s['worst']:.1f}",
                    "Std": f"{s['std']:.1f}",
                    "Ratio to Optimal": f"{s['ratio']:.3f}",
                    "Avg Improvement": f"{s['avg_improve']:.1%}",
                    "Avg Time (s)": f"{s['avg_time']:.2f}",
                    "Valid Runs": s["valid"],
                    "Total Runs": s["total"],
                    "Dijkstra Optimal": f"{opt_cost:.1f}",
                }
            )

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Results table saved to {filename}")


def fixed_param_robustness(terrain, routes, seeds):
    """Run SA with fixed default parameters across all routes — pure robustness test."""
    route_data = {}  # route_name -> {opt_cost, ratios_per_seed}
    for route_name, start, goal in routes:
        _, opt_cost = dijkstra(terrain, start, goal)
        results = Parallel(n_jobs=-1)(
            delayed(run_single_trial)(
                terrain,
                start,
                goal,
                DEFAULT_T0,
                DEFAULT_ALPHA,
                DEFAULT_ITERS,
                seed,
            )
            for seed in seeds
        )
        ratios = []
        for r in results:
            if r["valid"] and opt_cost > 0:
                ratios.append(r["cost"] / opt_cost)
        route_data[route_name] = {"opt_cost": opt_cost, "ratios": ratios}

    # Cross-route summary (only routes with valid ratios)
    reachable = {k: v for k, v in route_data.items() if v["ratios"]}
    all_ratios = [r for d in reachable.values() for r in d["ratios"]]

    if not all_ratios:
        return route_data

    route_means = [np.mean(d["ratios"]) for d in reachable.values()]
    cv_across_routes = (
        np.std(route_means) / np.mean(route_means)
        if len(route_means) > 1
        else 0
    )
    seed_cvs = [
        np.std(d["ratios"]) / np.mean(d["ratios"])
        for d in reachable.values()
        if len(d["ratios"]) > 1 and np.mean(d["ratios"]) > 0
    ]
    avg_seed_cv = np.mean(seed_cvs) if seed_cvs else 0

    # Save per-seed results to CSV
    per_seed_rows = []
    for route_name_key, d in route_data.items():
        for i, ratio in enumerate(d["ratios"]):
            per_seed_rows.append(
                {
                    "route": route_name_key,
                    "seed": seeds[i],
                    "dijkstra_optimal": d["opt_cost"],
                    "cost_ratio": ratio,
                }
            )
    with open("robustness_fixed_params_per_seed.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["route", "seed", "dijkstra_optimal", "cost_ratio"]
        )
        writer.writeheader()
        writer.writerows(per_seed_rows)
    print("  Per-seed results saved to robustness_fixed_params_per_seed.csv")

    # Save cross-route summary to CSV
    summary_rows = []
    for route_name_key, d in reachable.items():
        ratios = d["ratios"]
        avg = np.mean(ratios)
        std = np.std(ratios)
        cv = std / avg if avg > 0 else float("inf")
        summary_rows.append(
            {
                "route": route_name_key,
                "dijkstra_optimal": d["opt_cost"],
                "avg_ratio": avg,
                "std_ratio": std,
                "cv": cv,
                "best_ratio": min(ratios),
                "worst_ratio": max(ratios),
                "valid_runs": len(ratios),
                "total_runs": len(seeds),
            }
        )
    # Append a summary row
    summary_rows.append(
        {
            "route": "ALL_ROUTES",
            "dijkstra_optimal": "",
            "avg_ratio": np.mean(all_ratios),
            "std_ratio": np.std(all_ratios),
            "cv": (
                np.std(all_ratios) / np.mean(all_ratios)
                if np.mean(all_ratios) > 0
                else ""
            ),
            "best_ratio": min(all_ratios),
            "worst_ratio": max(all_ratios),
            "valid_runs": len(all_ratios),
            "total_runs": len(seeds) * len(routes),
        }
    )
    summary_fieldnames = [
        "route",
        "dijkstra_optimal",
        "avg_ratio",
        "std_ratio",
        "cv",
        "best_ratio",
        "worst_ratio",
        "valid_runs",
        "total_runs",
    ]
    with open("robustness_fixed_params_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print("  Summary saved to robustness_fixed_params_summary.csv")

    # Save cross-route comparison to CSV
    with open("robustness_fixed_params_cross_route.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"metric": "overall_avg_ratio", "value": np.mean(all_ratios)},
                {"metric": "cv_across_routes", "value": cv_across_routes},
                {"metric": "avg_cv_across_seeds", "value": avg_seed_cv},
                {
                    "metric": "route_choice_dominant",
                    "value": cv_across_routes > avg_seed_cv,
                },
            ]
        )
    print(
        "  Cross-route comparison saved to robustness_fixed_params_cross_route.csv"
    )

    # Boxplot: only reachable routes
    reachable_routes = [(n, s, g) for n, s, g in routes if n in reachable]
    fig, ax = plt.subplots(figsize=(max(6, len(reachable_routes) * 2), 5))
    data = [route_data[name]["ratios"] for name, _, _ in reachable_routes]
    labels = [name for name, _, _ in reachable_routes]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels)
    colors = [f"C{i}" for i in range(len(routes))]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(
        y=1.0, color="red", linestyle="--", alpha=0.5, label="Dijkstra optimal"
    )
    ax.set_ylabel("Cost / Dijkstra Optimal")
    ax.set_title(
        f"Algorithm Robustness Across Routes "
        f"(T0={DEFAULT_T0}, α={DEFAULT_ALPHA}, iter={DEFAULT_ITERS})"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("robustness_fixed_params.png", dpi=150)
    print(
        "  Fixed-parameter robustness plot saved to robustness_fixed_params.png"
    )

    # Convergence curves (one SA run per seed per route, snapshots every 200 iters)
    all_route_curves = {}
    for route_name, start, goal in routes:
        opt_cost = route_data[route_name]["opt_cost"]
        if not math.isfinite(opt_cost) or opt_cost <= 0:
            continue
        curves = []
        for seed in seeds:
            curves.append(
                convergence_curve(terrain, start, goal, opt_cost, seed)
            )
        all_route_curves[route_name] = curves
    plot_convergence(all_route_curves, filename="convergence_all_routes.png")

    return route_data


def robustness_test(terrain, start, goal, route_name, seeds):
    _, opt_cost = dijkstra(terrain, start, goal)

    # parameter sweep
    all_stats = []
    all_raw = []
    for param_name, values in [
        ("T0", [2000, 4000, 8000]),
        ("alpha", [0.995, 0.998, 0.999]),
        ("iterations", [1000, 2000, 4000]),
    ]:
        stats, raw = sweep_one_param(
            terrain, start, goal, opt_cost, param_name, values, seeds
        )
        all_stats += stats
        all_raw += raw

    safe_name = route_name.replace(" ", "_")
    plot_parameter_boxplots(
        all_raw, route_name, filename=f"boxplots_{safe_name}.png"
    )

    return opt_cost, all_stats


if __name__ == "__main__":
    FILE_NAME = "DHM25_subset_2.asc"
    X, Y, Z = load_elevation_data(FILE_NAME)
    terrain = TerrainGraph(FILE_NAME)

    routes = [
        (
            "Chuenihorn (short)",
            terrain.coords_to_rowcol(780692.4, 204899.9),
            terrain.coords_to_rowcol(780743.8, 206892.8),
        ),
        (
            "Schollberg (long)",
            terrain.coords_to_rowcol(781802.1, 205187.5),
            terrain.coords_to_rowcol(784502, 205753.2),
        ),
        (
            "Sulzfluh (long)",
            terrain.coords_to_rowcol(781802, 205188),
            terrain.coords_to_rowcol(782548, 209637),
        ),
    ]

    seeds = list(range(42, 52))  # 10 seeds

    # ── Step 1: Fixed-parameter robustness (convergence + cross-route boxplots) ──
    fixed_param_robustness(terrain, routes, seeds)

    # ── Step 2: Parameter sweep per route (per-parameter boxplots) ──
    all_route_stats = {}
    for name, start, goal in routes:
        opt_cost, stats = robustness_test(terrain, start, goal, name, seeds)
        all_route_stats[name] = (opt_cost, stats)

    save_results_table(all_route_stats)
