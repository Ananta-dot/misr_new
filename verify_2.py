#!/usr/bin/env python3
"""
Plot and verify a MISR instance (given as H/V sequences in JSON)
using Gurobi for LP and ILP, and matplotlib for visualization.

Usage:
    python plot_and_verify_misr_instance.py --json instance.json \
        --time_limit_ilp 5 --threads 8 --save_plot rects_n18.png
"""

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None
    # We'll error later with a clearer message


Seq = List[int]
Rect = Tuple[Tuple[int, int], Tuple[int, int]]  # ((x1,x2),(y1,y2)), half-open


@dataclass
class Instance:
    n: int
    H: Seq
    V: Seq


# ---------------------------------------------------------
# H/V helpers: spans -> rectangles
# ---------------------------------------------------------

def seq_spans(seq: Seq) -> List[Tuple[int, int]]:
    """
    For a sequence of length 2n with labels 1..n each appearing exactly twice,
    return spans [lo, hi] (hi exclusive) for each label i=1..n.
    seq positions are 0..2n-1, so spans live in [0,2n].
    """
    first: Dict[int, int] = {}
    second: Dict[int, int] = {}
    for idx, lab in enumerate(seq):
        if lab not in first:
            first[lab] = idx
        else:
            second[lab] = idx

    if not first:
        return []

    n = max(first.keys())
    spans: List[Tuple[int, int]] = []
    for lab in range(1, n + 1):
        if lab not in second:
            raise ValueError(f"Label {lab} does not appear exactly twice in sequence")
        p1 = first[lab]
        p2 = second[lab]
        lo = min(p1, p2)
        hi = max(p1, p2) + 1  # half-open
        spans.append((lo, hi))
    return spans


def build_rects(H: Seq, V: Seq) -> List[Rect]:
    """
    Build axis-aligned rectangles from H/V.
    We interpret indices as grid lines; spans become half-open intervals.
    """
    X = seq_spans(H)
    Y = seq_spans(V)
    if len(X) != len(Y):
        raise ValueError("H and V encode different numbers of rectangles")
    rects: List[Rect] = []
    for (x1, x2), (y1, y2) in zip(X, Y):
        rects.append(((x1, x2), (y1, y2)))
    return rects


# ---------------------------------------------------------
# Gurobi models: LP + ILP with point-stabbing constraints
# ---------------------------------------------------------

def build_cell_rects(rects: List[Rect], grid_size: int) -> Dict[Tuple[int, int], List[int]]:
    """
    Map each grid cell (i,j) to the list of rectangles that cover it.
    Grid cells are [i,i+1) x [j,j+1) for i,j in 0..grid_size-1.
    """
    from collections import defaultdict

    cell_rects: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for ridx, rect in enumerate(rects):
        (x1, x2), (y1, y2) = rect
        x1c = max(0, min(grid_size, x1))
        x2c = max(0, min(grid_size, x2))
        y1c = max(0, min(grid_size, y1))
        y2c = max(0, min(grid_size, y2))
        for i in range(x1c, x2c):
            for j in range(y1c, y2c):
                cell_rects[(i, j)].append(ridx)
    return cell_rects


def solve_lp_and_ilp(
    rects: List[Rect],
    grid_size: int,
    time_limit_ilp: float,
    threads: int,
) -> Tuple[float, np.ndarray, float]:
    """
    Solve:
      - LP relaxation (pure: no presolve/cuts) with variables x_r in [0,1]
      - ILP (binary z_r) with normal Gurobi presolve/cuts
    Constraints: for every grid cell, sum_{r covering cell} x_r <= 1.

    Returns: (lp_obj, frac_x, ilp_obj)
    """
    if gp is None:
        raise RuntimeError(
            "gurobipy is not available. Install Gurobi and gurobipy, "
            "and ensure your license is set up."
        )

    n = len(rects)
    cell_rects = build_cell_rects(rects, grid_size)

    # --- LP: pure relaxation, close to theoretical LP ---
    m_lp = gp.Model("misr_lp")
    m_lp.Params.OutputFlag = 0
    # Turn off presolve and cuts to keep LP as weak as the theoretical one
    m_lp.Params.Presolve = 0
    m_lp.Params.Cuts = 0
    m_lp.Params.CliqueCuts = 0
    m_lp.Params.CoverCuts = 0
    m_lp.Params.FlowCoverCuts = 0
    if threads > 0:
        m_lp.Params.Threads = threads

    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

    for (i, j), rect_ids in cell_rects.items():
        m_lp.addConstr(
            gp.quicksum(x[r] for r in rect_ids) <= 1.0,
            name=f"cell_{i}_{j}",
        )

    m_lp.setObjective(gp.quicksum(x[r] for r in range(n)), GRB.MAXIMIZE)
    m_lp.optimize()

    if m_lp.Status != GRB.OPTIMAL:
        raise RuntimeError(f"LP not optimal, status {m_lp.Status}")

    frac_x = np.array([x[r].X for r in range(n)], dtype=float)
    lp_obj = float(m_lp.ObjVal)

    # --- ILP: binary variables, allow Gurobi to use its full power ---
    m_ilp = gp.Model("misr_ilp")
    m_ilp.Params.OutputFlag = 0
    if time_limit_ilp > 0:
        m_ilp.Params.TimeLimit = time_limit_ilp
    if threads > 0:
        m_ilp.Params.Threads = threads

    z = m_ilp.addVars(n, vtype=GRB.BINARY, name="z")

    for (i, j), rect_ids in cell_rects.items():
        m_ilp.addConstr(
            gp.quicksum(z[r] for r in rect_ids) <= 1.0,
            name=f"cell_{i}_{j}",
        )

    m_ilp.setObjective(gp.quicksum(z[r] for r in range(n)), GRB.MAXIMIZE)
    m_ilp.optimize()

    if m_ilp.Status == GRB.TIME_LIMIT and m_ilp.SolCount == 0:
        raise RuntimeError("ILP hit time limit with no feasible solution")
    if m_ilp.SolCount == 0:
        raise RuntimeError(f"ILP has no feasible solution, status {m_ilp.Status}")

    ilp_obj = float(m_ilp.ObjVal)

    return lp_obj, frac_x, ilp_obj


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def plot_rects(
    rects: List[Rect],
    grid_size: int,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(6, 6))

    for idx, rect in enumerate(rects, start=1):
        (x1, x2), (y1, y2) = rect
        w = x2 - x1
        h = y2 - y1
        patch = Rectangle(
            (x1, y1),
            w,
            h,
            fill=False,
            linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(
            x1 + 0.5 * w,
            y1 + 0.5 * h,
            str(idx),
            ha="center",
            va="center",
            fontsize=7,
        )

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"MISR rectangles (n={len(rects)})")

    # Light grid for cells
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(which="both", linestyle=":", linewidth=0.5)

    # If you prefer origin at top-left, uncomment:
    # ax.invert_yaxis()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot and verify a MISR instance using Gurobi."
    )
    p.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to JSON file containing keys: n, H, V (and optionally gap, lp, ilp).",
    )
    p.add_argument(
        "--time_limit_ilp",
        type=float,
        default=5.0,
        help="Time limit (seconds) for ILP solve (0 = no limit).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of Gurobi threads (0 = Gurobi default).",
    )
    p.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="Optional path to save the rectangle plot as an image.",
    )
    p.add_argument(
        "--no_show",
        action="store_true",
        help="Do not display the plot window (useful on headless servers).",
    )
    return p.parse_args()


def load_instance(path: str) -> Instance:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n"])
    H = list(map(int, data["H"]))
    V = list(map(int, data["V"]))

    if len(H) != 2 * n or len(V) != 2 * n:
        raise ValueError(
            f"Expected H and V length 2n={2*n}, got len(H)={len(H)}, len(V)={len(V)}"
        )

    return Instance(n=n, H=H, V=V)


def main() -> None:
    args = parse_args()

    inst = load_instance(args.json)
    print(f"Loaded instance: n={inst.n}, len(H)=len(V)={len(inst.H)}")

    grid_size = len(inst.H)
    rects = build_rects(inst.H, inst.V)

    # Plot
    plot_rects(
        rects,
        grid_size=grid_size,
        save_path=args.save_plot,
        show=not args.no_show,
    )

    # Solve LP + ILP
    lp_val, frac_x, ilp_val = solve_lp_and_ilp(
        rects,
        grid_size=grid_size,
        time_limit_ilp=args.time_limit_ilp,
        threads=args.threads,
    )
    gap = lp_val / ilp_val if ilp_val > 0 else float("nan")

    print("\n=== Verification results ===")
    print(f"LP optimum     = {lp_val:.6f}")
    print(f"ILP optimum    = {ilp_val:.6f}")
    print(f"Integrality gap (LP/ILP) = {gap:.6f}")
    print("\nFractional x (first 20):")
    print(frac_x[:20])


if __name__ == "__main__":
    main()
