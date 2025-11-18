from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

# Type aliases
Seq = List[int]  # sequence of labels 1..n, each appearing exactly twice
Rect = Tuple[Tuple[int, int], Tuple[int, int]]  # ((x1,x2),(y1,y2)), half-open [x1,x2) x [y1,y2)


@dataclass
class Instance:
    H: Seq
    V: Seq


@dataclass
class Individual:
    inst: Instance
    lp: float
    ilp: float
    gap: float
    frac_x: np.ndarray


# ----------------------------------------------------------------------
# H/V encoding helpers
# ----------------------------------------------------------------------


def seq_spans(seq: Seq) -> List[Tuple[int, int]]:
    """
    For a sequence of length 2n with labels 1..n each appearing exactly twice,
    return for each label i its span [lo, hi] in *grid index* coordinates,
    where lo = min(pos1,pos2), hi = max(pos1,pos2)+1 (half-open).
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
        p1 = first[lab]
        p2 = second[lab]
        lo = min(p1, p2)
        hi = max(p1, p2) + 1  # half-open interval
        spans.append((lo, hi))
    return spans


def build_rects(H: Seq, V: Seq) -> List[Rect]:
    """
    Build axis-aligned rectangles from H/V sequences.
    We treat sequence positions as grid lines; spans become half-open intervals over grid cells.
    """
    X = seq_spans(H)
    Y = seq_spans(V)
    assert len(X) == len(Y), "H and V must encode the same number of rectangles"
    rects: List[Rect] = []
    for (x1, x2), (y1, y2) in zip(X, Y):
        rects.append(((x1, x2), (y1, y2)))
    return rects


# ----------------------------------------------------------------------
# Seed generators
# ----------------------------------------------------------------------


def random_seq(n: int, rng: random.Random) -> Seq:
    labels = list(range(1, n + 1)) * 2
    rng.shuffle(labels)
    return labels


def random_instance(n: int, rng: random.Random) -> Instance:
    return Instance(H=random_seq(n, rng), V=random_seq(n, rng))


def clique_like_instance(n: int, rng: random.Random) -> Instance:
    """
    Build a highly overlapping instance: nested intervals in H and V,
    with different nest orders to induce dense intersections.
    """
    L = 2 * n
    H = [0] * L
    V = [0] * L

    order_H = list(range(1, n + 1))
    order_V = list(range(1, n + 1))
    rng.shuffle(order_H)
    rng.shuffle(order_V)

    # Nested in H
    for pos, lab in enumerate(order_H):
        H[pos] = lab
        H[L - 1 - pos] = lab

    # Nested in V (different order)
    for pos, lab in enumerate(order_V):
        V[pos] = lab
        V[L - 1 - pos] = lab

    return Instance(H=H, V=V)


try:
    # Optional: if you have your own Chuzhoy seed generator, we hook into it
    from src.chuzhoy_seeds import seeds_for_n_from_chuzhoy  # type: ignore
except ImportError:
    seeds_for_n_from_chuzhoy = None  # type: ignore


def build_initial_population(
    n: int,
    pop_size: int,
    rng: random.Random,
    num_chu: int = 8,
    num_random: int = 24,
    num_clique: int = 8,
) -> List[Instance]:
    seeds: List[Instance] = []

    # 1) Chuzhoy-style seeds if available
    if seeds_for_n_from_chuzhoy is not None and num_chu > 0:
        hv_list = seeds_for_n_from_chuzhoy(n)
        hv_list = list(hv_list)
        rng.shuffle(hv_list)
        for H, V in hv_list[:num_chu]:
            seeds.append(Instance(H=list(H), V=list(V)))

    # 2) Pure random seeds
    for _ in range(num_random):
        seeds.append(random_instance(n, rng))

    # 3) Clique-ish seeds
    for _ in range(num_clique):
        seeds.append(clique_like_instance(n, rng))

    # Fill up if needed
    while len(seeds) < pop_size:
        seeds.append(random_instance(n, rng))

    return seeds[:pop_size]


# ----------------------------------------------------------------------
# Geometry / LP-ILP models
# ----------------------------------------------------------------------


def build_cell_rects(rects: List[Rect], grid_size: int) -> Dict[Tuple[int, int], List[int]]:
    """
    Build a mapping from grid cell (i,j) to list of rectangles that cover it.
    Grid cells are indexed 0..grid_size-1 in both x and y; rects are half-open [x1,x2)x[y1,y2).
    """
    from collections import defaultdict

    cell_rects: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for ridx, rect in enumerate(rects):
        (x1, x2), (y1, y2) = rect
        # Clamp to grid_size just in case
        x1_cl = max(0, min(grid_size, x1))
        x2_cl = max(0, min(grid_size, x2))
        y1_cl = max(0, min(grid_size, y1))
        y2_cl = max(0, min(grid_size, y2))
        for i in range(x1_cl, x2_cl):
            for j in range(y1_cl, y2_cl):
                cell_rects[(i, j)].append(ridx)
    return cell_rects


def solve_lp_and_ilp_for_rects(
    rects: List[Rect],
    grid_size: int,
    time_limit_ilp: float,
    threads: int,
) -> Tuple[float, np.ndarray, float]:
    """
    Solve the pure LP relaxation and the ILP for a given set of rectangles, using
    point-stabbing constraints on grid cells.
    Returns: (lp_obj, frac_x, ilp_obj)
    """
    if gp is None:
        raise RuntimeError("gurobipy is not available; install gurobi and gurobipy")

    n = len(rects)
    cell_rects = build_cell_rects(rects, grid_size)

    # ---- LP (pure relaxation, no Gurobi cuts/presolve) ----
    m_lp = gp.Model("misr_lp")
    m_lp.Params.OutputFlag = 0
    # Keep LP as close as possible to the theoretical one
    m_lp.Params.Presolve = 0
    m_lp.Params.Cuts = 0
    m_lp.Params.CliqueCuts = 0
    m_lp.Params.CoverCuts = 0
    m_lp.Params.FlowCoverCuts = 0
    if threads > 0:
        m_lp.Params.Threads = threads

    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

    # Constraints: for every grid cell, sum of x[r] over rects covering that cell <= 1
    for (i, j), rect_ids in cell_rects.items():
        m_lp.addConstr(gp.quicksum(x[r] for r in rect_ids) <= 1.0, name=f"cell_{i}_{j}")

    m_lp.setObjective(gp.quicksum(x[r] for r in range(n)), GRB.MAXIMIZE)
    m_lp.optimize()

    if m_lp.Status != GRB.OPTIMAL:
        raise RuntimeError(f"LP not optimal, status {m_lp.Status}")

    frac_x = np.array([x[r].X for r in range(n)], dtype=float)
    lp_obj = float(m_lp.ObjVal)

    # ---- ILP (binary variables, normal Gurobi machinery allowed) ----
    m_ilp = gp.Model("misr_ilp")
    m_ilp.Params.OutputFlag = 0
    if time_limit_ilp > 0:
        m_ilp.Params.TimeLimit = time_limit_ilp
    if threads > 0:
        m_ilp.Params.Threads = threads

    z = m_ilp.addVars(n, vtype=GRB.BINARY, name="z")

    for (i, j), rect_ids in cell_rects.items():
        m_ilp.addConstr(gp.quicksum(z[r] for r in rect_ids) <= 1.0, name=f"cell_{i}_{j}")

    m_ilp.setObjective(gp.quicksum(z[r] for r in range(n)), GRB.MAXIMIZE)
    m_ilp.optimize()

    if m_ilp.Status == GRB.TIME_LIMIT and not m_ilp.SolCount:
        raise RuntimeError("ILP hit time limit with no feasible solution")
    if m_ilp.SolCount == 0:
        raise RuntimeError(f"ILP has no feasible solution, status {m_ilp.Status}")

    ilp_obj = float(m_ilp.ObjVal)

    return lp_obj, frac_x, ilp_obj


# ----------------------------------------------------------------------
# LP-guided mutation
# ----------------------------------------------------------------------


def _positions_of_label(seq: Seq, label: int) -> Tuple[int, int]:
    pos = [i for i, lab in enumerate(seq) if lab == label]
    if len(pos) != 2:
        raise ValueError(f"Label {label} does not appear exactly twice")
    return pos[0], pos[1]


def _nudge_occurrence(seq: Seq, idx: int, rng: random.Random) -> None:
    L = len(seq)
    if L <= 1:
        return
    step = rng.choice([-2, -1, 1, 2])
    j = idx + step
    if j < 0:
        j = 0
    if j >= L:
        j = L - 1
    # swap
    seq[idx], seq[j] = seq[j], seq[idx]


def mutate_instance_hotspot_guided(
    inst: Instance,
    frac_x: np.ndarray,
    rng: random.Random,
    frac_low: float = 0.25,
    frac_high: float = 0.9,
    max_rects: int = 6,
) -> Instance:
    """
    LP-guided mutation:
    - pick rectangles with medium-large fractional weight
    - locally nudge their H/V occurrences
    """
    n = max(inst.H)  # assuming labels 1..n
    H = inst.H[:]
    V = inst.V[:]

    candidates = [i + 1 for i, x in enumerate(frac_x[:n])
                  if frac_low <= x <= frac_high]

    if not candidates:
        # fallback: just random swaps
        a, b = rng.sample(range(2 * n), 2)
        H[a], H[b] = H[b], H[a]
        a, b = rng.sample(range(2 * n), 2)
        V[a], V[b] = V[b], V[a]
        return Instance(H=H, V=V)

    rng.shuffle(candidates)
    candidates = candidates[:max_rects]

    for lab in candidates:
        h1, h2 = _positions_of_label(H, lab)
        v1, v2 = _positions_of_label(V, lab)

        _nudge_occurrence(H, rng.choice([h1, h2]), rng)
        _nudge_occurrence(V, rng.choice([v1, v2]), rng)

    return Instance(H=H, V=V)


# ----------------------------------------------------------------------
# Evaluation and search loop
# ----------------------------------------------------------------------


def evaluate_instance(
    inst: Instance,
    time_limit_ilp: float,
    threads: int,
) -> Individual:
    H, V = inst.H, inst.V
    assert len(H) == len(V), "H and V must have same length"
    grid_size = len(H)
    rects = build_rects(H, V)
    lp, frac_x, ilp = solve_lp_and_ilp_for_rects(rects, grid_size, time_limit_ilp, threads)
    if ilp <= 0:
        gap = 1.0
    else:
        gap = lp / ilp
    return Individual(inst=inst, lp=lp, ilp=ilp, gap=gap, frac_x=frac_x)


def search_high_gap_instances(
    n: int,
    rounds: int,
    pop_size: int,
    elite_frac: float,
    children_per_elite: int,
    time_limit_ilp: float,
    threads: int,
    rng: random.Random,
    num_chu: int,
    num_random: int,
    num_clique: int,
    target_gap: Optional[float] = None,
) -> Individual:
    population = build_initial_population(
        n=n,
        pop_size=pop_size,
        rng=rng,
        num_chu=num_chu,
        num_random=num_random,
        num_clique=num_clique,
    )

    best: Optional[Individual] = None

    for r in range(1, rounds + 1):
        t0 = time.time()
        evaluated: List[Individual] = []
        for inst in population:
            indiv = evaluate_instance(inst, time_limit_ilp=time_limit_ilp, threads=threads)
            evaluated.append(indiv)

        evaluated.sort(key=lambda x: x.gap, reverse=True)
        round_best = evaluated[0]
        t1 = time.time()

        if best is None or round_best.gap > best.gap:
            best = round_best

        print(
            f"[round {r:03d}] best_gap={round_best.gap:.4f} "
            f"(lp={round_best.lp:.2f}, ilp={round_best.ilp:.2f}) "
            f"global_best={best.gap:.4f} "
            f"| pop={len(population)} eval_time={t1 - t0:.2f}s"
        )

        if target_gap is not None and best.gap >= target_gap:
            print(f"Target gap {target_gap:.3f} reached; stopping.")
            break

        # ---- Build next population ----
        elite_count = max(1, int(math.ceil(elite_frac * len(evaluated))))
        elites = evaluated[:elite_count]

        new_population: List[Instance] = []

        # Keep elites and generate children via hotspot-guided mutation
        for indiv in elites:
            new_population.append(indiv.inst)  # keep parent
            for _ in range(children_per_elite):
                child_inst = mutate_instance_hotspot_guided(
                    indiv.inst, indiv.frac_x, rng
                )
                new_population.append(child_inst)

        # Diversity: add fresh random / clique seeds
        while len(new_population) < pop_size:
            u = rng.random()
            if u < 0.5:
                new_population.append(random_instance(n, rng))
            else:
                new_population.append(clique_like_instance(n, rng))

        population = new_population[:pop_size]

    assert best is not None
    return best


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search MISR instances with large LP/ILP gap.")

    p.add_argument("--n", type=int, default=18, help="Number of rectangles (labels).")
    p.add_argument("--rounds", type=int, default=30, help="Number of search rounds.")
    p.add_argument("--pop_size", type=int, default=64, help="Population size.")
    p.add_argument("--elite_frac", type=float, default=0.25, help="Elite fraction.")
    p.add_argument(
        "--children_per_elite",
        type=int,
        default=4,
        help="Number of LP-guided children per elite instance.",
    )

    p.add_argument(
        "--time_limit_ilp",
        type=float,
        default=5.0,
        help="Per-instance time limit for ILP (seconds, 0 = no limit).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of Gurobi threads (0 = Gurobi default).",
    )

    p.add_argument(
        "--num_chu",
        type=int,
        default=8,
        help="Number of Chuzhoy-style seeds in initial population (requires src.chuzhoy_seeds).",
    )
    p.add_argument(
        "--num_random",
        type=int,
        default=24,
        help="Number of purely random seeds in initial population.",
    )
    p.add_argument(
        "--num_clique",
        type=int,
        default=8,
        help="Number of clique-like seeds in initial population.",
    )

    p.add_argument(
        "--target_gap",
        type=float,
        default=None,
        help="Stop early if global best gap >= target_gap.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for RNG.",
    )

    p.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Optional path to write best instance as JSON.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)

    print(
        f"Starting MISR search: n={args.n}, rounds={args.rounds}, "
        f"pop_size={args.pop_size}, elite_frac={args.elite_frac}, "
        f"children_per_elite={args.children_per_elite}"
    )

    best = search_high_gap_instances(
        n=args.n,
        rounds=args.rounds,
        pop_size=args.pop_size,
        elite_frac=args.elite_frac,
        children_per_elite=args.children_per_elite,
        time_limit_ilp=args.time_limit_ilp,
        threads=args.threads,
        rng=rng,
        num_chu=args.num_chu,
        num_random=args.num_random,
        num_clique=args.num_clique,
        target_gap=args.target_gap,
    )

    print("\n=== Best instance found ===")
    print(f"gap = {best.gap:.6f}")
    print(f"lp  = {best.lp:.6f}")
    print(f"ilp = {best.ilp:.6f}")
    print(f"H   = {best.inst.H}")
    print(f"V   = {best.inst.V}")

    if args.out_json is not None:
        import json

        data = {
            "n": max(best.inst.H) if best.inst.H else 0,
            "gap": best.gap,
            "lp": best.lp,
            "ilp": best.ilp,
            "H": best.inst.H,
            "V": best.inst.V,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\nBest instance written to {args.out_json}")


if __name__ == "__main__":
    main()
