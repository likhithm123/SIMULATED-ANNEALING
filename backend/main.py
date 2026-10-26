import asyncio
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(
    title="Adaptive Simulated Annealing Platform",
    description="Real-Time Online Temperature Control & TSP Optimization Engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# TSP BENCHMARK GENERATOR & OPERATORS
# ============================================================

def generate_cities(n: int, seed: int = 42, distribution: str = "random") -> List[Dict[str, float]]:
    """
    Generate city coordinates on an 800x500 canvas.
    Supports random, clustered, circular, and grid distributions.
    """
    rng = np.random.default_rng(seed)
    cities: List[Dict[str, float]] = []

    if distribution == "circular":
        center_x, center_y = 400.0, 250.0
        radius = 180.0
        for i in range(n):
            angle = (2 * math.pi * i) / n + float(rng.uniform(-0.15, 0.15))
            r = radius + float(rng.uniform(-25, 25))
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            cities.append({"x": round(float(x), 2), "y": round(float(y), 2)})

    elif distribution == "clustered":
        num_clusters = max(2, min(5, n // 8))
        cluster_centers = [
            (float(rng.uniform(120, 680)), float(rng.uniform(100, 400)))
            for _ in range(num_clusters)
        ]
        for i in range(n):
            cx, cy = cluster_centers[i % num_clusters]
            x = np.clip(cx + rng.normal(0, 38), 50, 750)
            y = np.clip(cy + rng.normal(0, 32), 50, 450)
            cities.append({"x": round(float(x), 2), "y": round(float(y), 2)})

    elif distribution == "grid":
        # Manhattan-style grid lattice with organic jitter
        side = math.ceil(math.sqrt(n))
        xs = np.linspace(80, 720, side)
        ys = np.linspace(60, 440, side)
        grid_points = [(float(x), float(y)) for x in xs for y in ys]
        rng.shuffle(grid_points)
        for i in range(min(n, len(grid_points))):
            gx, gy = grid_points[i]
            x = float(np.clip(gx + rng.uniform(-10, 10), 50, 750))
            y = float(np.clip(gy + rng.uniform(-10, 10), 40, 460))
            cities.append({"x": round(x, 2), "y": round(y, 2)})

    elif distribution == "spiral":
        # Archimedean spiral: r = a * theta
        center_x, center_y = 400.0, 250.0
        max_turns = 3.2
        for i in range(n):
            theta = 0.5 + (max_turns * 2 * math.pi * (i + 1)) / n
            r = (190.0 * (i + 1)) / n + float(rng.uniform(-8, 8))
            x = center_x + r * math.cos(theta)
            y = center_y + (r * 0.75) * math.sin(theta)
            cities.append({
                "x": round(float(np.clip(x, 50, 750)), 2),
                "y": round(float(np.clip(y, 40, 460)), 2)
            })

    elif distribution == "berlin52":
        # Classic TSPLIB Berlin52 benchmark (canonical coordinates scaled to canvas)
        raw_berlin = [
            (565.0, 575.0), (25.0, 185.0), (345.0, 750.0), (945.0, 685.0), (845.0, 655.0),
            (880.0, 660.0), (25.0, 230.0), (525.0, 1000.0), (580.0, 1175.0), (650.0, 1130.0),
            (1605.0, 620.0), (1220.0, 580.0), (1465.0, 200.0), (1530.0, 5.0), (845.0, 680.0),
            (725.0, 370.0), (145.0, 665.0), (415.0, 635.0), (510.0, 875.0), (560.0, 365.0),
            (300.0, 465.0), (520.0, 585.0), (480.0, 415.0), (835.0, 625.0), (975.0, 580.0),
            (1215.0, 245.0), (1320.0, 315.0), (1250.0, 400.0), (660.0, 180.0), (410.0, 250.0),
            (420.0, 555.0), (575.0, 665.0), (1150.0, 1160.0), (700.0, 580.0), (685.0, 595.0),
            (685.0, 610.0), (770.0, 610.0), (795.0, 645.0), (720.0, 635.0), (760.0, 650.0),
            (475.0, 960.0), (95.0, 260.0), (875.0, 920.0), (700.0, 500.0), (555.0, 815.0),
            (830.0, 485.0), (1170.0, 65.0), (830.0, 610.0), (605.0, 625.0), (595.0, 360.0),
            (1340.0, 725.0), (1740.0, 245.0)
        ]
        min_x = min(c[0] for c in raw_berlin)
        max_x = max(c[0] for c in raw_berlin)
        min_y = min(c[1] for c in raw_berlin)
        max_y = max(c[1] for c in raw_berlin)
        count = min(n, len(raw_berlin)) if n > 0 else len(raw_berlin)
        for bx, by in raw_berlin[:count]:
            x = 80.0 + ((bx - min_x) / (max_x - min_x)) * 640.0
            y = 60.0 + ((by - min_y) / (max_y - min_y)) * 380.0
            cities.append({"x": round(float(x), 2), "y": round(float(y), 2)})

    else:  # "random"
        for _ in range(n):
            cities.append({
                "x": round(float(rng.uniform(60, 740)), 2),
                "y": round(float(rng.uniform(50, 450)), 2)
            })

    return cities


def distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def route_cost(route: List[int], cities: List[Dict[str, float]]) -> float:
    """Calculates the total closed-loop Euclidean perimeter of the tour."""
    if len(route) < 2:
        return 0.0
    total = 0.0
    n = len(route)
    for i in range(n):
        c1 = cities[route[i]]
        c2 = cities[route[(i + 1) % n]]
        total += math.hypot(c1["x"] - c2["x"], c1["y"] - c2["y"])
    return round(total, 2)


def two_opt_move(route: List[int]) -> Tuple[List[int], str]:
    """2-Opt: Reverses a sub-segment. Flips crossed edges to unknot tours."""
    n = len(route)
    if n < 4:
        return route.copy(), "2-opt"
    i, j = sorted(random.sample(range(n), 2))
    if j - i <= 1 or (i == 0 and j == n - 1):
        return swap_move(route)
    new_route = route.copy()
    new_route[i:j + 1] = reversed(new_route[i:j + 1])
    return new_route, "2-opt"


def swap_move(route: List[int]) -> Tuple[List[int], str]:
    """Swap: Swaps positions of two randomly chosen cities."""
    new_route = route.copy()
    i, j = random.sample(range(len(route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route, "swap"


def insert_move(route: List[int]) -> Tuple[List[int], str]:
    """Insert/Relocate: Removes a city and inserts it at another position."""
    new_route = route.copy()
    i, j = random.sample(range(len(route)), 2)
    city = new_route.pop(i)
    new_route.insert(j, city)
    return new_route, "insert"


def or_opt_move(route: List[int]) -> Tuple[List[int], str]:
    """
    Or-Opt: Relocates a connected block of 2 or 3 consecutive cities to another position in the tour.
    Crucial for preserving good local sub-paths while correcting higher-level sequencing.
    """
    n = len(route)
    if n < 6:
        return insert_move(route)

    seg_len = random.choice([2, 3])
    i = random.randint(0, n - seg_len)
    new_route = route.copy()
    segment = new_route[i : i + seg_len]
    del new_route[i : i + seg_len]

    j = random.randint(0, len(new_route))
    new_route[j:j] = segment
    return new_route, "or-opt"


def apply_neighborhood_operator(route: List[int], operator_weights: Optional[Dict[str, float]] = None) -> Tuple[List[int], str]:
    """
    Selects neighborhood operator based on dynamic weights.
    Defaults to 65% 2-opt, 15% or-opt, 12% insert, 8% swap.
    """
    if operator_weights is None:
        operator_weights = {"2-opt": 0.65, "or-opt": 0.15, "insert": 0.12, "swap": 0.08}

    ops = list(operator_weights.keys())
    weights = list(operator_weights.values())
    chosen_op = random.choices(ops, weights=weights, k=1)[0]

    if chosen_op == "2-opt":
        return two_opt_move(route)
    elif chosen_op == "or-opt":
        return or_opt_move(route)
    elif chosen_op == "insert":
        return insert_move(route)
    else:
        return swap_move(route)


# ============================================================
# MATHEMATICALLY ACCURATE TOUR DIVERSITY
# ============================================================

def get_tour_edges(route: List[int]) -> Set[frozenset]:
    """Extracts undirected edge pairs from a TSP tour."""
    n = len(route)
    return {frozenset((route[i], route[(i + 1) % n])) for i in range(n)}


def route_edge_difference(edges_a: Set[frozenset], edges_b: Set[frozenset], n: int) -> float:
    """
    Computes normalized Jaccard-like edge difference between two tours.
    Invariant to cyclic permutation and tour reversal.
    Returns 0.0 (identical tours) to 1.0 (completely distinct edges).
    """
    if n == 0:
        return 0.0
    common = len(edges_a.intersection(edges_b))
    diff = n - common
    return diff / n


def calculate_diversity(history_edges: List[Set[frozenset]], n_cities: int) -> float:
    """
    Computes average pairwise edge diversity among the last N accepted solutions.
    """
    m = len(history_edges)
    if m < 2 or n_cities == 0:
        return 0.5

    total_diff = 0.0
    comparisons = 0
    for i in range(m):
        for j in range(i + 1, m):
            total_diff += route_edge_difference(history_edges[i], history_edges[j], n_cities)
            comparisons += 1

    return round(total_diff / comparisons, 3) if comparisons > 0 else 0.0


# ============================================================
# ADAPTIVE TEMPERATURE CONTROLLER (FEEDBACK LOOP)
# ============================================================

class AdaptiveTemperatureController:
    """
    Online Feedback Controller for Simulated Annealing with Multi-Signal Priority Engine.
    Simultaneously analyzes 4 search dynamics:
      1. Acceptance Rate (W_acc)
      2. Improvement Rate (W_imp)
      3. Stagnation Count (S)
      4. Edge Diversity (D)

    Supports ablation modes:
      - 'full': all 4 signals evaluated with explicit multi-signal priority policy
      - 'no_diversity': ignores diversity metrics (D forced to nominal 0.50)
      - 'no_stagnation': ignores stagnation metrics (S forced to 0, no reheats)
      - 'no_acceptance': ignores acceptance rate adaptations (W_acc forced to nominal 0.25)
      - 'no_improvement': ignores fertile valley rapid cooling (W_imp forced to nominal 0.05)
      - 'acceptance_only': classical adaptive SA baseline responding purely to acceptance rate
      - 'standard_sa': fixed geometric cooling (no online feedback)
    """

    def __init__(
        self,
        initial_temperature: float = 600.0,
        min_temperature: float = 0.05,
        max_temperature: float = 1200.0,
        ablation_mode: str = "full",
        reheat_cooldown: int = 40,
        max_reheats: int = 3,
    ):
        self.temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.ablation_mode = ablation_mode
        self.reheat_cooldown = reheat_cooldown
        self.max_reheats = max_reheats

        # Multipliers
        self.reheat_structural_multiplier = 1.45
        self.reheat_local_multiplier = 1.30
        self.reheat_diversity_multiplier = 1.20
        self.boost_multiplier = 1.15
        self.cool_rapid_multiplier = 0.88
        self.cool_moderate_multiplier = 0.94
        self.cool_steady_multiplier = 0.985

        # Dynamic cooling rate and reheat budget tracking
        self.alpha = 0.995
        self.total_reheats = 0
        self.last_reheat_iteration = -100
        self.decision_history: List[Dict[str, Any]] = []

    def step_cool(self) -> float:
        """Step-wise cooling performed at each iteration step."""
        self.temperature = max(self.min_temperature, self.temperature * self.alpha)
        return self.temperature

    def update(
        self,
        iteration: int,
        window_acceptance_rate: float,
        window_improvement_rate: float,
        stagnation: int,
        diversity: float,
        current_best_cost: Optional[float] = None,
        total_iterations: int = 2000,
    ) -> Dict[str, Any]:
        old_temp = self.temperature
        reset_stagnation = False

        # Apply ablation filters
        w_acc = window_acceptance_rate if self.ablation_mode not in ("no_acceptance", "standard_sa") else 0.25
        w_imp = window_improvement_rate if self.ablation_mode not in ("no_improvement", "standard_sa", "acceptance_only") else 0.05
        stag = stagnation if self.ablation_mode not in ("no_stagnation", "standard_sa", "acceptance_only") else 0
        div = diversity if self.ablation_mode not in ("no_diversity", "standard_sa", "acceptance_only") else 0.50

        # Update outcome of previous decision if exists
        if self.decision_history and current_best_cost is not None:
            prev = self.decision_history[-1]
            if prev.get("cost_at_decision") is not None:
                cost_delta = round(current_best_cost - prev["cost_at_decision"], 2)
                prev["cost_delta"] = cost_delta
                if cost_delta < -0.01:
                    prev["outcome"] = f"IMPROVED ({cost_delta})"
                elif cost_delta > 0.01:
                    prev["outcome"] = f"REGRESSED (+{cost_delta})"
                else:
                    prev["outcome"] = "STABLE"

        can_reheat = (iteration - self.last_reheat_iteration) >= self.reheat_cooldown

        # --- ABLATION BASELINE: Standard SA (Blind geometric cooling) ---
        if self.ablation_mode == "standard_sa":
            self.alpha = 0.995
            self.temperature = max(self.min_temperature, self.temperature * 0.995)
            decision = "FIXED_COOL"
            mode = "BALANCED"
            reason = "Standard SA baseline: Fixed open-loop geometric cooling (alpha=0.995)."
            triggered_signals = []

        # --- ABLATION BASELINE: Acceptance-Only Adaptive SA ---
        elif self.ablation_mode == "acceptance_only":
            if w_acc < 0.15:
                self.temperature = min(self.max_temperature, self.temperature * 1.10)
                self.alpha = 0.996
                decision = "ACCEPTANCE_BOOST"
                mode = "EXPLORATION"
                reason = f"Acceptance rate low ({w_acc*100:.1f}% < 15%). Boosting temperature."
                triggered_signals = ["acceptance"]
            elif w_acc > 0.40:
                self.temperature = max(self.min_temperature, self.temperature * 0.92)
                self.alpha = 0.992
                decision = "ACCEPTANCE_COOL"
                mode = "EXPLOITATION"
                reason = f"Acceptance rate high ({w_acc*100:.1f}% > 40%). Quenching temperature."
                triggered_signals = ["acceptance"]
            else:
                self.temperature = max(self.min_temperature, self.temperature * 0.985)
                self.alpha = 0.995
                decision = "STEADY_COOL"
                mode = "BALANCED"
                reason = f"Acceptance rate nominal ({w_acc*100:.1f}% in [15%, 40%]). Steady cooling."
                triggered_signals = []

        # --- FULL MULTI-SIGNAL CONTROLLER (Explicit Priority Policy) ---
        else:
            is_deep_stagnation = (stag >= 100)
            is_diversity_collapsed = (div < 0.08)
            is_moderate_stagnation = (stag >= 50)
            is_freezing = (w_acc < 0.05 and self.temperature < (self.max_temperature * 0.70))
            is_fertile_valley = (w_imp > 0.12)
            is_overly_loose = (w_acc > 0.55)
            is_terminal_phase = (iteration > int(0.80 * total_iterations))

            # Terminal Phase Quench: Lock out reheats and freeze into minimum basin
            if is_terminal_phase:
                self.alpha = 0.991
                self.temperature = max(self.min_temperature, self.temperature * 0.95)
                decision = "TERMINAL_QUENCH"
                mode = "EXPLOITATION"
                reason = f"Final convergence phase ({iteration}/{total_iterations}): Reheats locked out to freeze into minimum."
                triggered_signals = ["convergence"]

            # Priority 1: Catastrophic Entrapment (Both Deep Stagnation AND Diversity Collapse)
            elif is_deep_stagnation and is_diversity_collapsed and can_reheat:
                if self.total_reheats < self.max_reheats:
                    self.temperature = min(self.max_temperature, max(self.temperature * self.reheat_structural_multiplier, 35.0))
                    self.alpha = 0.995
                    self.last_reheat_iteration = iteration
                    self.total_reheats += 1
                    reset_stagnation = True
                    decision = "STRUCTURAL_DIVERSIFICATION"
                    mode = "EXPLORATION"
                    reason = f"Dual Trigger: Stagnation={stag} steps & Diversity={div:.3f} < 0.08. Reheat escape #{self.total_reheats}/{self.max_reheats}."
                    triggered_signals = ["stagnation", "diversity"]
                else:
                    self.alpha = 0.992
                    decision = "BASIN_INTENSIFICATION"
                    mode = "EXPLOITATION"
                    reason = f"Reheat budget reached ({self.total_reheats}/{self.max_reheats}). Intensifying local exploitation."
                    triggered_signals = ["stagnation", "diversity"]

            # Priority 2: Deep Stagnation with Preserved Diversity (Local Minimum Basin)
            elif is_deep_stagnation and can_reheat:
                if self.total_reheats < self.max_reheats:
                    self.temperature = min(self.max_temperature, max(self.temperature * self.reheat_local_multiplier, 25.0))
                    self.alpha = 0.995
                    self.last_reheat_iteration = iteration
                    self.total_reheats += 1
                    reset_stagnation = True
                    decision = "LOCAL_MINIMA_REHEAT"
                    mode = "EXPLORATION"
                    reason = f"Stagnation={stag} steps in local basin. Reheat escape #{self.total_reheats}/{self.max_reheats}."
                    triggered_signals = ["stagnation"]
                else:
                    self.alpha = 0.992
                    decision = "BASIN_INTENSIFICATION"
                    mode = "EXPLOITATION"
                    reason = f"Reheat budget reached ({self.total_reheats}/{self.max_reheats}). Intensifying local exploitation."
                    triggered_signals = ["stagnation"]

            # Priority 3: Diversity Collapse at Moderate Stagnation (Cyclic Tour Loops)
            elif is_diversity_collapsed and is_moderate_stagnation and can_reheat:
                if self.total_reheats < self.max_reheats:
                    self.temperature = min(self.max_temperature, max(self.temperature * self.reheat_diversity_multiplier, 20.0))
                    self.alpha = 0.995
                    self.last_reheat_iteration = iteration
                    self.total_reheats += 1
                    reset_stagnation = True
                    decision = "DIVERSITY_REHEAT"
                    mode = "EXPLORATION"
                    reason = f"Diversity collapsed to {div:.3f} with stagnation={stag}. Reheat escape #{self.total_reheats}/{self.max_reheats}."
                    triggered_signals = ["diversity", "stagnation"]
                else:
                    self.alpha = 0.992
                    decision = "BASIN_INTENSIFICATION"
                    mode = "EXPLOITATION"
                    reason = f"Reheat budget reached ({self.total_reheats}/{self.max_reheats}). Intensifying local exploitation."
                    triggered_signals = ["diversity", "stagnation"]

            # Priority 4: Critical Freezing Risk (Low Acceptance)
            elif is_freezing:
                self.temperature = min(self.max_temperature, max(self.temperature * self.boost_multiplier, 15.0))
                self.alpha = 0.997
                decision = "THAW_BOOST"
                mode = "EXPLORATION"
                reason = f"Premature freeze: Acceptance={w_acc*100:.1f}% < 5%. Boosting temperature to restore mobility."
                triggered_signals = ["acceptance"]

            # Priority 5: Fertile Gradient Discovery (Rapid Progress in Attraction Basin)
            elif is_fertile_valley:
                self.temperature = max(self.min_temperature, self.temperature * self.cool_rapid_multiplier)
                self.alpha = 0.993
                decision = "RAPID_COOL"
                mode = "EXPLOITATION"
                reason = f"Fertile valley: Improvement rate={w_imp*100:.1f}% > 12%. Accelerating local exploitation."
                triggered_signals = ["improvement"]

            # Priority 6: High Random Drift (Overly Loose Search)
            elif is_overly_loose:
                self.temperature = max(self.min_temperature, self.temperature * self.cool_moderate_multiplier)
                self.alpha = 0.993
                decision = "MODERATE_COOL"
                mode = "EXPLOITATION"
                reason = f"Acceptance high ({w_acc*100:.1f}% > 55%). Cooling to reduce random Brownian walk."
                triggered_signals = ["acceptance"]

            # Priority 7: Nominal Balanced Cooling
            else:
                self.temperature = max(self.min_temperature, self.temperature * self.cool_steady_multiplier)
                self.alpha = 0.995
                decision = "STEADY_COOL"
                mode = "BALANCED"
                reason = f"Search dynamics inside nominal bounds (Acc: {w_acc*100:.1f}%, Imp: {w_imp*100:.1f}%, Div: {div:.2f}). Steady cooling."
                triggered_signals = []

        entry = {
            "iteration": iteration,
            "signals": {
                "acceptance_rate": round(window_acceptance_rate, 3),
                "improvement_rate": round(window_improvement_rate, 3),
                "stagnation": stagnation,
                "diversity": round(diversity, 3),
            },
            "triggered_signals": triggered_signals,
            "decision": decision,
            "mode": mode,
            "reason": reason,
            "old_temperature": round(old_temp, 2),
            "new_temperature": round(self.temperature, 2),
            "cost_at_decision": current_best_cost,
            "cost_delta": None,
            "outcome": "PENDING",
            "ablation_mode": self.ablation_mode,
            "reset_stagnation": reset_stagnation,
            "total_reheats": self.total_reheats,
            "alpha": round(self.alpha, 4),
        }
        self.decision_history.append(entry)
        return entry


# ============================================================
# SESSION CONTROLLER (PAUSE, RESUME, SPEED, STOP)
# ============================================================

class SessionContext:
    def __init__(self):
        self.is_paused = False
        self.is_stopped = False
        self.delay_seconds = 0.003
        self.step_requested = False

    def reset(self, delay_ms: float = 3.0):
        self.is_paused = False
        self.is_stopped = False
        self.delay_seconds = max(0.0, delay_ms / 1000.0)
        self.step_requested = False


# ============================================================
# ALGORITHM ENGINES
# ============================================================

async def run_conventional_sa(
    websocket: WebSocket,
    cities: List[Dict[str, float]],
    iterations: int,
    session: SessionContext,
    initial_route: Optional[List[int]] = None,
    initial_temp: float = 600.0,
    cooling_alpha: float = 0.995,
):
    """
    Conventional Simulated Annealing with fixed geometric cooling schedule:
    T_{k+1} = alpha * T_k.
    """
    n = len(cities)
    if initial_route is not None:
        route = initial_route.copy()
    else:
        route = list(range(n))
        random.shuffle(route)

    current_cost = route_cost(route, cities)
    best_route = route.copy()
    best_cost = current_cost

    temperature = initial_temp
    accepted_all = 0
    improvements_all = 0

    # Rolling window tracking
    WINDOW_SIZE = 50
    recent_accepts = []
    recent_improvements = []
    history_edges: List[Set[frozenset]] = [get_tour_edges(route)]

    stagnation = 0

    for iteration in range(1, iterations + 1):
        if session.is_stopped:
            break

        while session.is_paused and not session.step_requested and not session.is_stopped:
            await asyncio.sleep(0.05)
        session.step_requested = False

        candidate, op_name = apply_neighborhood_operator(route)
        candidate_cost = route_cost(candidate, cities)
        delta = candidate_cost - current_cost

        is_accepted = False
        is_improved = False

        # Metropolis-Hastings criterion
        if delta < 0 or (temperature > 1e-6 and random.random() < math.exp(-delta / temperature)):
            route = candidate
            current_cost = candidate_cost
            is_accepted = True
            accepted_all += 1
            history_edges.append(get_tour_edges(route))
            if len(history_edges) > 15:
                history_edges.pop(0)

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_route = candidate.copy()
                is_improved = True
                improvements_all += 1
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1

        recent_accepts.append(1 if is_accepted else 0)
        recent_improvements.append(1 if is_improved else 0)
        if len(recent_accepts) > WINDOW_SIZE:
            recent_accepts.pop(0)
            recent_improvements.pop(0)

        # Fixed geometric cooling
        temperature = max(0.01, temperature * cooling_alpha)

        # Send updates every 5 iterations or on completion
        if iteration % 5 == 0 or iteration == iterations:
            win_acc = sum(recent_accepts) / len(recent_accepts)
            win_imp = sum(recent_improvements) / len(recent_improvements)
            diversity = calculate_diversity(history_edges, n)

            await websocket.send_json({
                "type": "iteration",
                "algorithm": "Conventional SA",
                "iteration": iteration,
                "current_cost": current_cost,
                "best_cost": best_cost,
                "temperature": round(temperature, 2),
                "acceptance_rate": round(win_acc, 3),
                "improvement_rate": round(win_imp, 3),
                "cumulative_acceptance": round(accepted_all / iteration, 3),
                "stagnation": stagnation,
                "diversity": diversity,
                "mode": "FIXED COOLING",
                "decision": "FIXED_SCHEDULE",
                "reason": f"Fixed cooling schedule (alpha={cooling_alpha}). No feedback control.",
                "route": best_route,
                "operator": op_name,
            })

        if session.delay_seconds > 0:
            await asyncio.sleep(session.delay_seconds)

    await websocket.send_json({
        "type": "complete",
        "algorithm": "Conventional SA",
        "best_cost": best_cost,
        "route": best_route,
        "iterations_completed": iteration,
        "stagnation_events": 0,
    })


async def run_adaptive_sa(
    websocket: WebSocket,
    cities: List[Dict[str, float]],
    iterations: int,
    session: SessionContext,
    initial_route: Optional[List[int]] = None,
    initial_temp: float = 600.0,
    ablation_mode: str = "full",
):
    """
    Adaptive Simulated Annealing with online feedback control.
    Supports ablation modes: 'full', 'no_diversity', 'no_stagnation', 'no_acceptance', 'no_improvement', 'acceptance_only'.
    Continuously monitors search metrics and adjusts temperature & operator strategy.
    """
    n = len(cities)
    if initial_route is not None:
        route = initial_route.copy()
    else:
        route = list(range(n))
        random.shuffle(route)

    current_cost = route_cost(route, cities)
    best_route = route.copy()
    best_cost = current_cost

    controller = AdaptiveTemperatureController(
        initial_temperature=initial_temp,
        min_temperature=0.05,
        max_temperature=1200.0,
        ablation_mode=ablation_mode,
    )

    accepted_all = 0
    improvements_all = 0
    stagnation = 0
    total_reheats = 0

    WINDOW_SIZE = 50
    recent_accepts = []
    recent_improvements = []
    history_edges: List[Set[frozenset]] = [get_tour_edges(route)]

    operator_stats = {
        op: {"proposed": 0, "accepted": 0, "improved": 0}
        for op in ["2-opt", "or-opt", "insert", "swap"]
    }

    decision_data = {
        "decision": "INITIALIZING",
        "mode": "BALANCED",
        "reason": f"Starting adaptive optimization (ablation mode: {ablation_mode}).",
        "old_temperature": initial_temp,
        "new_temperature": initial_temp,
        "triggered_signals": [],
    }

    # Adaptive operator weights
    op_weights = {"2-opt": 0.65, "or-opt": 0.15, "insert": 0.12, "swap": 0.08}

    for iteration in range(1, iterations + 1):
        if session.is_stopped:
            break

        while session.is_paused and not session.step_requested and not session.is_stopped:
            await asyncio.sleep(0.05)
        session.step_requested = False

        candidate, op_name = apply_neighborhood_operator(route, op_weights)
        operator_stats[op_name]["proposed"] += 1
        candidate_cost = route_cost(candidate, cities)
        delta = candidate_cost - current_cost

        is_accepted = False
        is_improved = False

        # Metropolis-Hastings acceptance using dynamic temperature
        curr_t = max(controller.temperature, 1e-6)
        if delta < 0 or random.random() < math.exp(-delta / curr_t):
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
            else:
                stagnation += 1
        else:
            stagnation += 1

        recent_accepts.append(1 if is_accepted else 0)
        recent_improvements.append(1 if is_improved else 0)
        if len(recent_accepts) > WINDOW_SIZE:
            recent_accepts.pop(0)
            recent_improvements.pop(0)

        # Continuous step cooling governed by adaptive controller's alpha
        controller.step_cool()

        # Update controller epoch every 40 iterations
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
                total_iterations=iterations,
            )

            total_reheats = controller.total_reheats

            if decision_data.get("reset_stagnation"):
                stagnation = 0
                route = best_route.copy()
                current_cost = best_cost
                if decision_data["decision"] == "STRUCTURAL_DIVERSIFICATION":
                    # Perturb with Or-Opt / Insert on diversity collapse
                    op_weights = {"2-opt": 0.40, "or-opt": 0.30, "insert": 0.20, "swap": 0.10}
                else:
                    op_weights = {"2-opt": 0.55, "or-opt": 0.25, "insert": 0.12, "swap": 0.08}
            elif decision_data["decision"] == "TERMINAL_QUENCH":
                op_weights = {"2-opt": 0.88, "or-opt": 0.10, "insert": 0.02, "swap": 0.00}
            elif decision_data["mode"] == "EXPLOITATION":
                # In exploitation mode, emphasize 2-opt and or-opt untangling
                op_weights = {"2-opt": 0.80, "or-opt": 0.12, "insert": 0.05, "swap": 0.03}
            else:
                op_weights = {"2-opt": 0.65, "or-opt": 0.15, "insert": 0.12, "swap": 0.08}

        # Send updates every 5 iterations or at end
        if iteration % 5 == 0 or iteration == iterations:
            win_acc = sum(recent_accepts) / len(recent_accepts) if recent_accepts else 0.5
            win_imp = sum(recent_improvements) / len(recent_improvements) if recent_improvements else 0.1
            diversity = calculate_diversity(history_edges, n)

            await websocket.send_json({
                "type": "iteration",
                "algorithm": f"Adaptive SA ({ablation_mode})" if ablation_mode != "full" else "Adaptive Simulated Annealing",
                "ablation_mode": ablation_mode,
                "iteration": iteration,
                "current_cost": current_cost,
                "best_cost": best_cost,
                "temperature": round(controller.temperature, 2),
                "acceptance_rate": round(win_acc, 3),
                "improvement_rate": round(win_imp, 3),
                "cumulative_acceptance": round(accepted_all / iteration, 3),
                "stagnation": stagnation,
                "diversity": diversity,
                "mode": decision_data["mode"],
                "decision": decision_data["decision"],
                "reason": decision_data["reason"],
                "triggered_signals": decision_data.get("triggered_signals", []),
                "route": best_route,
                "operator": op_name,
                "total_reheats": total_reheats,
                "operator_stats": operator_stats,
            })

        if session.delay_seconds > 0:
            await asyncio.sleep(session.delay_seconds)

    await websocket.send_json({
        "type": "complete",
        "algorithm": f"Adaptive SA ({ablation_mode})" if ablation_mode != "full" else "Adaptive Simulated Annealing",
        "ablation_mode": ablation_mode,
        "best_cost": best_cost,
        "route": best_route,
        "iterations_completed": iteration,
        "stagnation_events": total_reheats,
        "decision_log": controller.decision_history,
        "operator_stats": operator_stats,
    })


async def run_benchmark_comparison(
    websocket: WebSocket,
    cities: List[Dict[str, float]],
    iterations: int,
    session: SessionContext,
):
    """
    Side-by-side benchmark comparison running Conventional SA then Adaptive SA
    on the EXACT SAME initial route and city topology.
    """
    n = len(cities)
    initial_route = list(range(n))
    random.shuffle(initial_route)

    # 1. Run Conventional SA
    await websocket.send_json({
        "type": "benchmark_phase",
        "phase": "conventional",
        "message": "Phase 1/2: Running Conventional SA benchmark...",
    })
    await run_conventional_sa(
        websocket=websocket,
        cities=cities,
        iterations=iterations,
        session=session,
        initial_route=initial_route,
    )

    if session.is_stopped:
        return

    await asyncio.sleep(0.5)

    # 2. Run Adaptive SA
    await websocket.send_json({
        "type": "benchmark_phase",
        "phase": "adaptive",
        "message": "Phase 2/2: Running Adaptive SA benchmark on the identical problem...",
    })
    await run_adaptive_sa(
        websocket=websocket,
        cities=cities,
        iterations=iterations,
        session=session,
        initial_route=initial_route,
    )


# ============================================================
# WEBSOCKET & REST API
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = SessionContext()
    running_task: Optional[asyncio.Task] = None

    try:
        while True:
            text = await websocket.receive_text()
            data = json.loads(text)
            action = data.get("action", "start")

            if action == "pause":
                session.is_paused = True
                await websocket.send_json({"type": "status", "status": "paused"})
                continue

            elif action == "resume":
                session.is_paused = False
                await websocket.send_json({"type": "status", "status": "running"})
                continue

            elif action == "step":
                session.step_requested = True
                continue

            elif action == "stop":
                session.is_stopped = True
                if running_task and not running_task.done():
                    running_task.cancel()
                await websocket.send_json({"type": "status", "status": "stopped"})
                continue

            elif action == "set_speed":
                delay_ms = float(data.get("delay_ms", 3.0))
                session.delay_seconds = max(0.0, delay_ms / 1000.0)
                continue

            elif action == "start":
                # Terminate any previously running task
                if running_task and not running_task.done():
                    session.is_stopped = True
                    running_task.cancel()
                    await asyncio.sleep(0.05)

                session.reset(delay_ms=float(data.get("delay_ms", 3.0)))

                algorithm = data.get("algorithm", "adaptive")
                city_count = int(data.get("cities", 50))
                iterations = int(data.get("iterations", 2000))
                seed = int(data.get("seed", random.randint(1, 100000)))
                distribution = data.get("distribution", "random")

                custom_coords = data.get("custom_cities")
                if custom_coords and len(custom_coords) >= 4:
                    cities = [{"x": float(c["x"]), "y": float(c["y"])} for c in custom_coords]
                else:
                    cities = generate_cities(city_count, seed=seed, distribution=distribution)

                # Send cities to frontend first
                await websocket.send_json({
                    "type": "cities",
                    "cities": cities,
                    "seed": seed,
                    "distribution": distribution,
                })

                if algorithm in ("conventional", "standard_sa"):
                    running_task = asyncio.create_task(
                        run_conventional_sa(websocket, cities, iterations, session)
                    )
                elif algorithm == "compare":
                    running_task = asyncio.create_task(
                        run_benchmark_comparison(websocket, cities, iterations, session)
                    )
                else:
                    # Accepts 'adaptive', 'full', or any ablation mode
                    ablation = "full" if algorithm in ("adaptive", "full") else algorithm
                    running_task = asyncio.create_task(
                        run_adaptive_sa(websocket, cities, iterations, session, ablation_mode=ablation)
                    )

    except WebSocketDisconnect:
        session.is_stopped = True
        if running_task and not running_task.done():
            running_task.cancel()
    except Exception as e:
        session.is_stopped = True
        if running_task and not running_task.done():
            running_task.cancel()
        print(f"WebSocket session error: {e}")


# ============================================================
# BENCHMARK SUITE & ABLATION ENDPOINTS
# ============================================================

from benchmark_runner import (
    run_multi_seed_benchmark,
    export_benchmark_to_csv,
    SCALED_BERLIN52_OPTIMAL,
)

LATEST_BENCHMARK_RESULTS: Optional[Dict[str, Any]] = None


@app.post("/api/benchmark/suite")
def trigger_benchmark_suite(payload: Dict[str, Any]):
    """
    Executes an automated multi-seed benchmark & ablation study.
    Runs identical initial tours and coordinates across seeds and algorithms.
    """
    global LATEST_BENCHMARK_RESULTS
    distribution = payload.get("distribution", "berlin52")
    city_count = int(payload.get("cities", 52))
    iterations = int(payload.get("iterations", 1500))
    seeds = payload.get("seeds", [42, 101, 777, 999, 1234])
    algorithms = payload.get("algorithms", [
        "standard_sa",
        "acceptance_only",
        "full",
        "no_diversity",
        "no_stagnation",
        "no_improvement",
    ])

    results = run_multi_seed_benchmark(
        distribution=distribution,
        city_count=city_count,
        iterations=iterations,
        seeds=seeds,
        algorithms=algorithms,
    )
    LATEST_BENCHMARK_RESULTS = results
    return results


@app.get("/api/benchmark/export_csv")
def export_benchmark_csv():
    """Exports the latest benchmark suite results as a downloadable CSV."""
    global LATEST_BENCHMARK_RESULTS
    if not LATEST_BENCHMARK_RESULTS:
        LATEST_BENCHMARK_RESULTS = run_multi_seed_benchmark(
            distribution="berlin52",
            city_count=52,
            iterations=1200,
            seeds=[42, 101, 777],
        )

    csv_data = export_benchmark_to_csv(LATEST_BENCHMARK_RESULTS)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=benchmark_ablation_results.csv"}
    )


@app.get("/api/benchmark/latest")
def get_latest_benchmark():
    global LATEST_BENCHMARK_RESULTS
    return LATEST_BENCHMARK_RESULTS or {"status": "no_benchmark_run_yet"}


# ============================================================
# STATIC FILES & WEB GUI
# ============================================================

@app.get("/", response_class=HTMLResponse)
def serve_home():
    """Serves the interactive web interface directly."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Frontend index.html not found</h1>", status_code=404)


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "Adaptive Simulated Annealing Platform",
        "version": "2.2.0",
        "algorithms": [
            "full",
            "acceptance_only",
            "standard_sa",
            "no_diversity",
            "no_stagnation",
            "no_acceptance",
            "no_improvement",
            "compare",
        ],
        "distributions": ["random", "clustered", "circular", "grid", "spiral", "berlin52"],
        "operators": ["2-opt", "or-opt", "insert", "swap"],
    }


@app.get("/api/presets")
def get_presets():
    return {
        "distributions": [
            {"id": "random", "name": "Random Scatter", "description": "Uniform random 2D spatial distribution"},
            {"id": "clustered", "name": "Clustered Logistics Hubs", "description": "Multi-center Gaussian clusters representing freight depots"},
            {"id": "circular", "name": "Circular Ring Topology", "description": "Concentric perturbed circular ring"},
            {"id": "grid", "name": "Manhattan Grid Lattice", "description": "Regular coordinate grid with minor jitter"},
            {"id": "spiral", "name": "Archimedean Spiral", "description": "Continuous spiral curve challenging local search"},
            {"id": "berlin52", "name": "Berlin52 (TSPLIB)", "description": "Canonical TSPLIB 52-city benchmark instance (Known Opt: 7610.94)"},
        ],
        "algorithms": [
            {"id": "adaptive", "name": "Full Adaptive SA (All Signals)", "description": "Online controller using acceptance, improvement, stagnation, & diversity"},
            {"id": "acceptance_only", "name": "Adaptive Cooling (Acceptance Only)", "description": "Classical adaptive SA baseline adjusting purely to acceptance rate"},
            {"id": "standard_sa", "name": "Standard SA (Geometric Cooling)", "description": "Blind open-loop exponential schedule (alpha=0.995)"},
            {"id": "no_diversity", "name": "Ablation: No Diversity Signal", "description": "Diversity feedback disabled"},
            {"id": "no_stagnation", "name": "Ablation: No Stagnation Reheats", "description": "Stagnation-triggered reheats disabled"},
            {"id": "no_acceptance", "name": "Ablation: No Acceptance Feedback", "description": "Acceptance thaw/cool disabled"},
            {"id": "no_improvement", "name": "Ablation: No Fertile Rapid Cool", "description": "Improvement gradient bursts disabled"},
            {"id": "compare", "name": "Comparative Benchmark (Standard vs Full)", "description": "Direct side-by-side execution on identical seed"},
        ],
        "operators": ["2-opt", "or-opt", "insert", "swap"],
        "berlin52_optimum": SCALED_BERLIN52_OPTIMAL,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8008))
    print(f"Starting Adaptive Simulated Annealing Platform on http://127.0.0.1:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
