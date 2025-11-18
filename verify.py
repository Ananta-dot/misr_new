#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot and summarize MISR elites saved by mistr_runner.py

- Loads a .pkl (list of (ratio, H, V) tuples)
- Prints a summary table
- Optionally recomputes exact LP/ILP with Gurobi for verification
- Plots rectangles for the top-K instances to PNG files

Usage examples:
  python3.11 plot_misr_elites.py --pkl misr_elites.pkl --top 12
  python3.11 plot_misr_elites.py --pkl misr_elites_n14_r3.pkl --top 8 --recompute
  python3.11 plot_misr_elites.py --pkl misr_elites_n20_final.pkl --out plots_n20 --dpi 200 --grid --annot
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -----------------------------
# Basic MISR helpers (match mistr_runner.py)
# -----------------------------
Seq = List[int]
Rect = Tuple[Tuple[int,int], Tuple[int,int]]  # ((x1,x2),(y1,y2))

def seq_spans(seq: Seq) -> List[Tuple[int, int]]:
    """Return [l_i, r_i] (indices of the two occurrences) for labels i=1..n."""
    first = {}
    spans = {}
    for idx, lab in enumerate(seq):
        if lab not in first:
            first[lab] = idx
        else:
            spans[lab] = (first[lab], idx)
    n = max(seq) if seq else 0
    return [spans[i] for i in range(1, n + 1)]

def build_rects(H: Seq, V: Seq) -> List[Rect]:
    X = seq_spans(H); Y = seq_spans(V)
    rects = []
    for (x1,x2),(y1,y2) in zip(X,Y):
        if x1>x2: x1,x2=x2,x1
        if y1>y2: y1,y2=y2,y1
        rects.append(((x1,x2),(y1,y2)))
    return rects

# -----------------------------
# Optional: exact LP/ILP recomputation (needs Gurobi)
# -----------------------------
def _grid_points(rects: List[Rect]):
    xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
    ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
    return [(x,y) for x in xs for y in ys]

def _covers_grid_closed(rects: List[Rect], pts):
    covers = []
    for (x,y) in pts:
        S=[]
        for i,((x1,x2),(y1,y2)) in enumerate(rects):
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                S.append(i)
        covers.append(S)
    return covers

def recompute_lp_ilp(rects: List[Rect], grb_threads: int = 0):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return None  # Gurobi not available

    pts = _grid_points(rects)
    covers = _covers_grid_closed(rects, pts)

    # LP
    m_lp = gp.Model("misr_lp_plot"); m_lp.setParam('OutputFlag', 0)
    if grb_threads > 0: m_lp.setParam('Threads', grb_threads)
    n = len(rects)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x')
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1.0)
    m_lp.optimize()
    lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else 0.0

    # ILP
    m_ilp = gp.Model("misr_ilp_plot"); m_ilp.setParam('OutputFlag', 0)
    if grb_threads > 0: m_ilp.setParam('Threads', grb_threads)
    y = m_ilp.addVars(n, vtype=GRB.BINARY, name='y')
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1.0)
    m_ilp.optimize()
    ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else 0.0

    ratio = (lp/ilp) if ilp > 0 else 0.0
    return lp, ilp, ratio

# -----------------------------
# Plotting
# -----------------------------
def plot_instance(H: Seq, V: Seq, ratio_hint: Optional[float],
                  out_path: str, title: Optional[str] = None,
                  show_grid: bool = False, annotate_labels: bool = False,
                  dpi: int = 160, transparent: bool = False):
    rects = build_rects(H, V)
    n = max(H) if H else 0

    fig, ax = plt.subplots(figsize=(6,6), dpi=dpi)

    # visually separate overlapping rectangles a bit
    eps = 0.05

    # build a repeating color cycle
    import itertools
    # cycle through tab20 then fallback to default cycle
    colors = plt.cm.tab20.colors if hasattr(plt.cm, "tab20") else plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    color_iter = itertools.cycle(colors)

    for lab, ((x1,x2),(y1,y2)) in enumerate(rects, start=1):
        w = (x2 - x1) if x2 > x1 else 0.8
        h = (y2 - y1) if y2 > y1 else 0.8
        # Rectangle takes (x, y, width, height)
        c = next(color_iter)
        patch = Rectangle((x1+eps, y1+eps), max(w-2*eps, 0.6), max(h-2*eps, 0.6),
                          fill=True, alpha=0.18, edgecolor=c, linewidth=1.2, facecolor=c)
        ax.add_patch(patch)
        if annotate_labels:
            ax.text((x1+x2)/2, (y1+y2)/2, str(lab),
                    ha='center', va='center', fontsize=8, color='k')

    L = len(H)
    ax.set_xlim(-0.5, L-0.5)
    ax.set_ylim(-0.5, L-0.5)
    ax.set_aspect('equal', adjustable='box')

    if show_grid:
        ax.set_xticks(range(0, L))
        ax.set_yticks(range(0, L))
        ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # title
    t = title if title else f"MISR rectangles (n={n})"
    if ratio_hint is not None:
        t += f" | ratio≈{ratio_hint:.3f}"
    ax.set_title(t, fontsize=12)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, transparent=transparent)
    plt.close(fig)

# -----------------------------
# Main CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, default="misr_elites.pkl", help="Path to elites pickle")
    ap.add_argument("--top", type=int, default=10, help="How many top instances to output")
    ap.add_argument("--out", type=str, default="plots", help="Output directory for images")
    ap.add_argument("--dpi", type=int, default=160, help="PNG DPI")
    ap.add_argument("--grid", action="store_true", help="Draw grid lines")
    ap.add_argument("--annot", action="store_true", help="Annotate rectangle labels in the plot")
    ap.add_argument("--recompute", action="store_true", help="Recompute exact LP/ILP with Gurobi")
    ap.add_argument("--threads", type=int, default=0, help="Gurobi threads (0 = default)")
    args = ap.parse_args()

    # Load pickle
    with open(args.pkl, "rb") as f:
        elites = pickle.load(f)

    if not elites:
        print("No elites found in pickle.")
        return

    # Normalize structure: list of (ratio, H, V)
    norm = []
    for item in elites:
        if isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], (float, int)):
            ratio, H, V = item
        else:
            # Fallback if a different structure was saved
            # Try to detect by fields
            try:
                ratio, H, V = float(item[0]), list(item[1]), list(item[2])
            except Exception:
                raise ValueError("Unexpected pickle structure; expected list of (ratio, H, V)")
        norm.append((float(ratio), H, V))

    # Sort by ratio desc
    norm.sort(key=lambda x: -x[0])

    K = min(args.top, len(norm))
    print(f"Loaded {len(norm)} elites from {args.pkl}. Showing top {K}:\n")

    print(f"{'rank':>4}  {'n':>3}  {'ratio_hint':>10}  {'recomp(LP,ILP,ratio)':>26}")
    print("-"*50)
    for rank, (ratio_hint, H, V) in enumerate(norm[:K], start=1):
        n = max(H) if H else 0
        recomputed = None
        if args.recompute:
            recomputed = recompute_lp_ilp(build_rects(H,V), grb_threads=args.threads)
        if recomputed is None:
            print(f"{rank:>4}  {n:>3}  {ratio_hint:>10.3f}  {'(skip)':>26}")
        else:
            lp, ilp, rr = recomputed
            print(f"{rank:>4}  {n:>3}  {ratio_hint:>10.3f}  ({lp:6.2f},{ilp:5.2f},{rr:5.3f})")

    # Plot and save PNGs
    base = os.path.splitext(os.path.basename(args.pkl))[0]
    for rank, (ratio_hint, H, V) in enumerate(norm[:K], start=1):
        n = max(H) if H else 0
        out_path = os.path.join(args.out, f"{base}_rank{rank}_n{n}_r{ratio_hint:.3f}.png")
        plot_instance(H, V, ratio_hint, out_path,
                      title=f"{base} | rank {rank}",
                      show_grid=args.grid,
                      annotate_labels=args.annot,
                      dpi=args.dpi,
                      transparent=False)
    print(f"\nSaved {K} plots to: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
