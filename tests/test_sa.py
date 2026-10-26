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
# 4. ADAPTIVE CONTROLLER SENSOR TESTS
# ============================================================

def test_adaptive_controller_reheat():
    controller = AdaptiveTemperatureController(initial_temperature=500.0)
    # Stagnation >= 100 should trigger REHEAT
    res = controller.update(
        iteration=100,
        window_acceptance_rate=0.20,
        window_improvement_rate=0.02,
        stagnation=105,
        diversity=0.35,
    )
    assert res["decision"] == "REHEAT"
    assert res["mode"] == "EXPLORATION"
    assert controller.temperature > 500.0


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


# ============================================================
# 5. REST API ENDPOINT TESTS
# ============================================================

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "berlin52" in data["distributions"]
    assert "or-opt" in data["operators"]


def test_presets_endpoint():
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.json()
    assert len(data["distributions"]) >= 6
    assert "2-opt" in data["operators"]


def test_frontend_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
