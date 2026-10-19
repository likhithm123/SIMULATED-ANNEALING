# Adaptive Simulated Annealing (ASA) Platform

An interactive real-time visual laboratory and optimization engine demonstrating **Adaptive Simulated Annealing with Online Feedback Temperature Control** applied to the Traveling Salesperson Problem (TSP).

---

## 🌟 Why This Project Exists

In optimization and computer science, finding the shortest route connecting $N$ cities (**Traveling Salesperson Problem**) is NP-hard. For just 50 cities, there are over $10^{64}$ possible routes—more than the number of grains of sand on Earth.

### The Metal Analogy: Annealing
- In blacksmithing, heating metal softens it and allows atoms to freely reorganize into crystal lattices. As it cools slowly, it hardens into an ultra-tough, defect-free structure.
- In optimization algorithms, **High Temperature** lets the algorithm explore freely and take risks (even accepting worse moves). **Low Temperature** forces it to settle down and exploit the best local solutions.

### The Flaw of Conventional Simulated Annealing
Standard Simulated Annealing uses a **fixed, blind cooling formula**:
$$T_{k+1} = \alpha \times T_k \quad (\text{e.g., } \alpha = 0.995)$$

Once temperature drops, the algorithm loses its kinetic energy. If it gets trapped inside a **local minimum** (a dead-end valley that is better than its immediate neighbors, but far worse than the global best), **it can never escape**. It is frozen forever.

### The Solution: Adaptive Feedback Control
Adaptive Simulated Annealing acts like a **smart thermostat with sensors**:
1. **Stagnation Sensor**: Tracks how many iterations have passed without discovering a better route. If trapped for $\ge 100$ steps, it triggers a **Reheat event**, injecting kinetic energy so the route can uncross knotted loops.
2. **Sliding-Window Acceptance Sensor**: Measures recent acceptance rate. If acceptance collapses below 5%, it boosts temperature to avoid premature freezing.
3. **Edge Diversity Sensor**: Measures the topological difference between recent solutions using Jaccard edge sets. If the search degenerates into repeating the same loop, it diversifies.
4. **Opportunistic Cooling**: When a fertile valley is discovered (high improvement rate), it cools rapidly to lock in the discovery.

---

## 🚀 Quick Start (Running Locally)

### 1. Launch the Server
Execute the launch script in your terminal:
```bash
./run.sh
```
Or manually run with the virtual environment:
```bash
source backend/venv/bin/activate
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8008 --reload
```

### 2. Open the Web Dashboard
Open your browser and navigate to:
👉 **[http://127.0.0.1:8008](http://127.0.0.1:8008)**

---

## 🎮 Platform Features & Interactive Controls

1. **Algorithm Modes**:
   - **Adaptive Simulated Annealing**: Real-time feedback control, automatic reheating, and adaptive neighborhood operators.
   - **Conventional Simulated Annealing**: Standard fixed geometric cooling schedule.
   - **Side-by-Side Benchmark**: Executes Conventional SA and Adaptive SA sequentially on the **exact same city topology and starting route**, then provides an automated scoreboard comparison.
2. **City Topologies**:
   - **Random Scatter**: Uniform distribution across the 2D space.
   - **Clustered Logistics Hubs**: Distinct clusters mimicking real-world regional freight depots.
   - **Circular Ring**: Cities distributed along a noisy circular perimeter.
3. **Custom City Designer**:
   - Click anywhere on the route canvas to insert custom cities before starting a run.
4. **Execution Controls**:
   - Real-time pause, step-by-step frame advance, and dynamic speed slider (0ms instant to 30ms slow motion).
5. **Real-Time Visualizations**:
   - Dynamic canvas with glowing tour perimeters and node labels.
   - Convergence chart comparing cost reduction over time.
   - Live AI decision log detailing the rationale behind every cooling or reheating action.

---

## 📂 Project Architecture

```
adaptive-sa/
├── backend/
│   ├── main.py              # FastAPI server, WebSocket endpoints, ASA Controller & TSP logic
│   ├── requirements.txt     # Python dependencies (fastapi, uvicorn, numpy, websockets)
│   └── venv/                # Local virtual environment
├── frontend/
│   └── index.html           # Interactive HTML5 Canvas + Chart.js glassmorphism web GUI
├── run.sh                   # One-click startup script
└── README.md                # Documentation and theoretical overview
```

---

## 🔬 Mathematical Details

- **Distance Metric**: Euclidean metric $d(a, b) = \sqrt{(x_a - x_b)^2 + (y_a - y_b)^2}$
- **Metropolis-Hastings Probability**:
  $$P(\text{accept } s' \mid s, T) = \begin{cases} 1 & \text{if } \Delta E \le 0 \\ e^{-\Delta E / T} & \text{if } \Delta E > 0 \end{cases}$$
- **Tour Edge Diversity**:
  Given edge set $E_A$ and $E_B$ for tours of length $n$:
  $$\text{Diversity}(A, B) = \frac{n - |E_A \cap E_B|}{n}$$
