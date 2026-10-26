import math
import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from main import (
    app,
    generate_cities,
    distance,
    route_cost,
    two_opt_move,
    swap_move,
    insert_move,
    or_opt_move,
    apply_neighborhood_operator,
    get_tour_edges,
    route_edge_difference,
    calculate_diversity,
    AdaptiveTemperatureController,
)


# ============================================================
# 1. CITY GENERATOR TESTS
# ============================================================

@pytest.mark.parametrize("dist", ["random", "clustered", "circular", "grid", "spiral", "berlin52"])
def test_generate_cities(dist):
    n = 20 if dist != "berlin52" else 52
    cities = generate_cities(n=n, seed=42, distribution=dist)
    assert len(cities) == n
    for c in cities:
        assert "x" in c and "y" in c
        assert 0 <= c["x"] <= 800
        assert 0 <= c["y"] <= 500


def test_distance():
    p1 = {"x": 0.0, "y": 0.0}
    p2 = {"x": 3.0, "y": 4.0}
    assert math.isclose(distance(p1, p2), 5.0)


def test_route_cost_square():
    # 4 points forming a 10x10 square
    cities = [
        {"x": 0.0, "y": 0.0},
        {"x": 10.0, "y": 0.0},
        {"x": 10.0, "y": 10.0},
        {"x": 0.0, "y": 10.0},
    ]
    route = [0, 1, 2, 3]
    # perimeter = 10 + 10 + 10 + 10 = 40.0
    assert math.isclose(route_cost(route, cities), 40.0, rel_tol=1e-2)


# ============================================================
# 2. NEIGHBORHOOD OPERATOR TESTS
# ============================================================

def test_neighborhood_operators_preserve_permutation():
    original = list(range(12))
    
    # 2-opt
    for _ in range(10):
        new_r, op = two_opt_move(original)
        assert len(new_r) == 12
        assert sorted(new_r) == original
        assert op in ("2-opt", "swap")

    # Swap
    for _ in range(10):
        new_r, op = swap_move(original)
        assert len(new_r) == 12
        assert sorted(new_r) == original
        assert op == "swap"

    # Insert
    for _ in range(10):
        new_r, op = insert_move(original)
        assert len(new_r) == 12
        assert sorted(new_r) == original
        assert op == "insert"

    # Or-opt
    for _ in range(10):
        new_r, op = or_opt_move(original)
        assert len(new_r) == 12
        assert sorted(new_r) == original
        assert op == "or-opt"


def test_apply_neighborhood_operator():
    original = list(range(10))
    for _ in range(30):
        candidate, op = apply_neighborhood_operator(original)
        assert sorted(candidate) == original
        assert op in ("2-opt", "or-opt", "insert", "swap")


# ============================================================
# 3. TOPOLOGICAL EDGE DIVERSITY TESTS
# ============================================================

def test_tour_edges_invariance():
    # Tour 1
    t1 = [0, 1, 2, 3, 4]
    # Tour 2: Cyclic shift of Tour 1
    t2 = [2, 3, 4, 0, 1]
    # Tour 3: Complete reversal of Tour 1
    t3 = [4, 3, 2, 1, 0]

    e1 = get_tour_edges(t1)
    e2 = get_tour_edges(t2)
    e3 = get_tour_edges(t3)

    assert e1 == e2
    assert e1 == e3
    assert route_edge_difference(e1, e2, 5) == 0.0
    assert route_edge_difference(e1, e3, 5) == 0.0


def test_edge_diversity_disjoint():
    # Two distinct tours
    t1 = [0, 1, 2, 3]
    t2 = [0, 2, 1, 3]
    e1 = get_tour_edges(t1)
    e2 = get_tour_edges(t2)
    diff = route_edge_difference(e1, e2, 4)
    assert 0.0 < diff <= 1.0


# ============================================================
# 4. ADAPTIVE CONTROLLER SENSOR & PRIORITY TESTS
# ============================================================

def test_adaptive_controller_local_minima_reheat():
    controller = AdaptiveTemperatureController(initial_temperature=500.0)
    # Stagnation >= 100 with healthy diversity should trigger LOCAL_MINIMA_REHEAT
    res = controller.update(
        iteration=100,
        window_acceptance_rate=0.20,
        window_improvement_rate=0.02,
        stagnation=105,
        diversity=0.35,
        current_best_cost=1200.0,
    )
    assert res["decision"] == "LOCAL_MINIMA_REHEAT"
    assert res["mode"] == "EXPLORATION"
    assert "stagnation" in res["triggered_signals"]
    assert controller.temperature > 500.0


def test_adaptive_controller_structural_diversification():
    controller = AdaptiveTemperatureController(initial_temperature=500.0)
    # Both Stagnation >= 100 and Diversity < 0.08 triggers STRUCTURAL_DIVERSIFICATION
    res = controller.update(
        iteration=100,
        window_acceptance_rate=0.15,
        window_improvement_rate=0.01,
        stagnation=110,
        diversity=0.04,
        current_best_cost=1500.0,
    )
    assert res["decision"] == "STRUCTURAL_DIVERSIFICATION"
    assert res["mode"] == "EXPLORATION"
    assert "stagnation" in res["triggered_signals"]
    assert "diversity" in res["triggered_signals"]
    assert controller.temperature > 500.0 * 1.40


def test_adaptive_controller_diversity_reheat():
    controller = AdaptiveTemperatureController(initial_temperature=500.0)
    # Moderate stagnation (>=50) + Diversity < 0.08 triggers DIVERSITY_REHEAT
    res = controller.update(
        iteration=100,
        window_acceptance_rate=0.25,
        window_improvement_rate=0.03,
        stagnation=60,
        diversity=0.05,
        current_best_cost=1400.0,
    )
    assert res["decision"] == "DIVERSITY_REHEAT"
    assert res["mode"] == "EXPLORATION"
    assert "diversity" in res["triggered_signals"]


def test_adaptive_controller_thaw():
    controller = AdaptiveTemperatureController(initial_temperature=300.0)
    # Very low acceptance rate (<5%) should trigger THAW_BOOST
    res = controller.update(
        iteration=40,
        window_acceptance_rate=0.02,
        window_improvement_rate=0.01,
        stagnation=10,
        diversity=0.25,
    )
    assert res["decision"] == "THAW_BOOST"
    assert res["mode"] == "EXPLORATION"
    assert controller.temperature > 300.0


def test_adaptive_controller_rapid_cool():
    controller = AdaptiveTemperatureController(initial_temperature=600.0)
    # High improvement rate (>12%) should trigger RAPID_COOL
    res = controller.update(
        iteration=40,
        window_acceptance_rate=0.30,
        window_improvement_rate=0.18,
        stagnation=0,
        diversity=0.40,
    )
    assert res["decision"] == "RAPID_COOL"
    assert res["mode"] == "EXPLOITATION"
    assert controller.temperature < 600.0


def test_controller_decision_outcome_tracking():
    controller = AdaptiveTemperatureController(initial_temperature=500.0)
    # Decision 1 at cost 1000
    controller.update(
        iteration=40,
        window_acceptance_rate=0.02,
        window_improvement_rate=0.01,
        stagnation=10,
        diversity=0.30,
        current_best_cost=1000.0,
    )
    assert len(controller.decision_history) == 1
    assert controller.decision_history[0]["outcome"] == "PENDING"

    # Decision 2 at improved cost 950 -> Decision 1 outcome should be updated to IMPROVED
    controller.update(
        iteration=80,
        window_acceptance_rate=0.30,
        window_improvement_rate=0.15,
        stagnation=0,
        diversity=0.30,
        current_best_cost=950.0,
    )
    assert "IMPROVED" in controller.decision_history[0]["outcome"]
    assert controller.decision_history[0]["cost_delta"] == -50.0


# ============================================================
# 5. ABLATION MODES TESTS
# ============================================================

def test_ablation_no_diversity():
    controller = AdaptiveTemperatureController(ablation_mode="no_diversity")
    # Even if diversity is 0.02, it should NOT trigger DIVERSITY_REHEAT
    res = controller.update(
        iteration=40,
        window_acceptance_rate=0.25,
        window_improvement_rate=0.02,
        stagnation=60,
        diversity=0.02,
    )
    assert res["decision"] != "DIVERSITY_REHEAT"


def test_ablation_no_stagnation():
    controller = AdaptiveTemperatureController(ablation_mode="no_stagnation")
    # Even if stagnation is 150, it should NOT trigger REHEAT
    res = controller.update(
        iteration=100,
        window_acceptance_rate=0.25,
        window_improvement_rate=0.02,
        stagnation=150,
        diversity=0.30,
    )
    assert "REHEAT" not in res["decision"]


def test_ablation_acceptance_only():
    controller = AdaptiveTemperatureController(ablation_mode="acceptance_only")
    res_boost = controller.update(
        iteration=40,
        window_acceptance_rate=0.05,
        window_improvement_rate=0.00,
        stagnation=120,
        diversity=0.01,
    )
    assert res_boost["decision"] == "ACCEPTANCE_BOOST"


def test_ablation_standard_sa():
    controller = AdaptiveTemperatureController(initial_temperature=600.0, ablation_mode="standard_sa")
    res = controller.update(
        iteration=40,
        window_acceptance_rate=0.01,
        window_improvement_rate=0.25,
        stagnation=200,
        diversity=0.01,
    )
    assert res["decision"] == "FIXED_COOL"
    assert math.isclose(controller.temperature, 600.0 * 0.995, rel_tol=1e-3)


# ============================================================
# 6. BENCHMARK RUNNER & REST API TESTS
# ============================================================

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "berlin52" in data["distributions"]
    assert "or-opt" in data["operators"]
    assert "full" in data["algorithms"]
    assert "acceptance_only" in data["algorithms"]


def test_presets_endpoint():
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.json()
    assert len(data["distributions"]) >= 6
    assert "2-opt" in data["operators"]
    assert "berlin52_optimum" in data


def test_frontend_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_benchmark_suite_api():
    payload = {
        "distribution": "berlin52",
        "cities": 52,
        "iterations": 200,
        "seeds": [42, 101],
        "algorithms": ["standard_sa", "full"],
    }
    response = client.post("/api/benchmark/suite", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "standard_sa" in data["summary"]
    assert "full" in data["summary"]
    assert data["summary"]["full"]["mean_cost"] > 0
    assert len(data["raw_runs"]) == 4  # 2 algorithms x 2 seeds


def test_benchmark_export_csv():
    response = client.get("/api/benchmark/export_csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "Algorithm,InitialCost,BestCost" in response.text


def test_adaptive_controller_step_cool():
    controller = AdaptiveTemperatureController(initial_temperature=600.0)
    controller.alpha = 0.995
    t1 = controller.step_cool()
    assert math.isclose(t1, 600.0 * 0.995, rel_tol=1e-3)
    t2 = controller.step_cool()
    assert t2 < t1


def test_adaptive_controller_terminal_quench():
    controller = AdaptiveTemperatureController(initial_temperature=50.0)
    # At iteration 1700 out of 2000 (85%), even with deep stagnation and low diversity,
    # it must enter TERMINAL_QUENCH to freeze into the minimum basin rather than reheating
    res = controller.update(
        iteration=1700,
        window_acceptance_rate=0.01,
        window_improvement_rate=0.0,
        stagnation=150,
        diversity=0.02,
        total_iterations=2000,
    )
    assert res["decision"] == "TERMINAL_QUENCH"
    assert res["mode"] == "EXPLOITATION"
    assert res["reset_stagnation"] is False
    assert controller.total_reheats == 0


def test_adaptive_controller_reheat_capping():
    controller = AdaptiveTemperatureController(initial_temperature=300.0, max_reheats=2)
    # Reheat 1
    r1 = controller.update(iteration=40, window_acceptance_rate=0.2, window_improvement_rate=0.01, stagnation=120, diversity=0.4)
    assert r1["decision"] == "LOCAL_MINIMA_REHEAT"
    assert r1["reset_stagnation"] is True
    assert controller.total_reheats == 1

    # Reheat 2
    r2 = controller.update(iteration=90, window_acceptance_rate=0.2, window_improvement_rate=0.01, stagnation=120, diversity=0.4)
    assert r2["decision"] == "LOCAL_MINIMA_REHEAT"
    assert controller.total_reheats == 2

    # Reheat 3 should be capped -> BASIN_INTENSIFICATION
    r3 = controller.update(iteration=140, window_acceptance_rate=0.2, window_improvement_rate=0.01, stagnation=120, diversity=0.4)
    assert r3["decision"] == "BASIN_INTENSIFICATION"
    assert r3["mode"] == "EXPLOITATION"
    assert controller.total_reheats == 2
    assert r3["reset_stagnation"] is False
