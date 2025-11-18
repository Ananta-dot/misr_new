#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
misr_qd_seek.py
Quality–Diversity (MAP-Elites) seeker for MISR instances.

- Loads top elites from your PKL files
- Builds a MAP-Elites archive using behavior descriptors that correlate with higher LP/ILP gaps:
    BD1: number of distinct interior midpoints with coverage >= 2   (spread of overlaps)
    BD2: max point load (or optionally span-entropy; see comment)

- Mutates with structural moves (nest / weave / interleave + usual swaps/moves)
- Keeps the best in each (BD1, BD2) bin by true ratio (LP/ILP using open-midpoints)
- Periodically dumps a checkpoint PKL of archive bests you can feed back to PatternBoost

Designed to complement your current pipeline; it does not require torch.
"""

import argparse, glob, os, pickle, random, time, math
from typing import List, Tuple, Dict, Optional

# ---------- Types & basic utils ----------
Seq = List[int]
Instance = Tuple[Seq, Seq]
Rect = Tuple[Tuple[int,int], Tuple[int,int]]  # ((x1,x2),(y1,y2))

def canonicalize(H: Seq, V: Seq) -> Instance:
    order=[]; seen=set()
    for x in H:
        if x not in seen:
            order.append(x); seen.add(x)
    rel = {old:new for new,old in enumerate(order,1)}
    return [rel[x] for x in H], [rel[x] for x in V]

def seq_spans(seq: Seq) -> List[Tuple[int, int]]:
    first={}; spans={}
    for idx, lab in enumerate(seq):
        if lab not in first: first[lab]=idx
        else: spans[lab]=(first[lab], idx)
    n = max(seq) if seq else 0
    return [spans[i] for i in range(1, n+1)]

def build_rects(H: Seq, V: Seq) -> List[Rect]:
    X = seq_spans(H); Y = seq_spans(V)
    rects=[]
    for (x1,x2),(y1,y2) in zip(X,Y):
        if x1>x2: x1,x2=x2,x1
        if y1>y2: y1,y2=y2,y1
        rects.append(((x1,x2),(y1,y2)))
    return rects

# ---------- Candidate point sets (open interior) ----------
def open_midpoints_points(rects: List[Rect]):
    xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
    ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
    midx = [ (xs[i]+xs[i+1])/2.0 for i in range(len(xs)-1) ] if len(xs)>1 else []
    midy = [ (ys[i]+ys[i+1])/2.0 for i in range(len(ys)-1) ] if len(ys)>1 else []
    pts = [(x,y) for x in midx for y in midy]
    for (x1,x2),(y1,y2) in rects:
        pts.append(((x1+x2)/2.0, (y1+y2)/2.0))
    pts = list({(float(round(x,6)), float(round(y,6))) for (x,y) in pts})
    return pts

def covers_open(rects: List[Rect], pts):
    C=[]
    for (x,y) in pts:
        S=[]
        for i,((x1,x2),(y1,y2)) in enumerate(rects):
            if (x1 < x < x2) and (y1 < y < y2):
                S.append(i)
        C.append(S)
    return C

# ---------- Exact LP/ILP via Gurobi ----------
def solve_lp_ilp(rects: List[Rect], pts, threads=0):
    import gurobipy as gp
    from gurobipy import GRB
    covers = covers_open(rects, pts)

    m_lp = gp.Model("misr_lp"); m_lp.setParam('OutputFlag', 0)
    if threads>0: m_lp.setParam('Threads', threads)
    n = len(rects)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S: m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m_lp.optimize()
    lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else 0.0

    m_ilp = gp.Model("misr_ilp"); m_ilp.setParam('OutputFlag', 0)
    if threads>0: m_ilp.setParam('Threads', threads)
    y = m_ilp.addVars(n, vtype=GRB.BINARY)
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S: m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1)
    m_ilp.optimize()
    ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else 0.0

    return lp, ilp, covers

def score_and_bd(H: Seq, V: Seq, threads=0):
    rects = build_rects(H,V)
    pts = open_midpoints_points(rects)
    lp, ilp, covers = solve_lp_ilp(rects, pts, threads=threads)
    ratio = (lp/ilp) if ilp>0 else 0.0
    # Behavior descriptors:
    # BD1: number of points with coverage >= 2
    bd1 = sum(1 for S in covers if len(S) >= 2)
    # BD2: max point load (max |S|)  (alternatively: span-entropy)
    bd2 = max((len(S) for S in covers), default=0)
    return ratio, bd1, bd2

# ---------- Structural mutations ----------
def neighbors(H: Seq, V: Seq, rng: random.Random, k=200):
    def apply_nest(seq: Seq):
        labs=list(set(seq))
        if len(labs)<3: return seq[:]
        s = rng.randint(3, min(6, len(labs)))
        chosen = rng.sample(labs, s)
        rest = [x for x in seq if x not in chosen]
        block = chosen[:] + list(reversed(chosen))
        pos = rng.randrange(len(rest)+1)
        return rest[:pos] + block + rest[pos:]

    def apply_interleave(seq: Seq):
        labs=list(set(seq))
        if len(labs)<2: return seq[:]
        pairs = rng.randint(1, min(4, len(labs)//2))
        chosen = rng.sample(labs, pairs*2)
        rest = [x for x in seq if x not in chosen]
        block=[]
        for i in range(0,len(chosen),2):
            a,b=chosen[i],chosen[i+1]
            block += [a,b,a,b]
        pos = rng.randrange(len(rest)+1)
        out = rest[:pos] + block + rest[pos:]
        # repair counts
        cnt={}
        for x in out: cnt[x]=cnt.get(x,0)+1
        for a in range(1, max(seq)+1):
            cnt.setdefault(a,0)
            while cnt[a]<2: out.append(a); cnt[a]+=1
        return out[:2*max(seq)]

    def apply_weave(seq: Seq):
        labs=list(set(seq))
        if len(labs)<3: return seq[:]
        k = rng.randint(3, min(7, len(labs)))
        chosen = rng.sample(labs, k)
        out = [x for x in seq if x not in chosen]
        # place first occurrences left-to-right
        for a in chosen:
            p = rng.randrange(len(out)+1)
            out.insert(p, a)
        # second occurrences roughly mirrored
        for a in reversed(chosen):
            p = len(out)-rng.randrange(len(out)+1)
            out.insert(min(len(out), p), a)
        # repair counts
        cnt={}
        for x in out: cnt[x]=cnt.get(x,0)+1
        for a in list(cnt.keys()):
            while cnt[a]>2:
                out.remove(a); cnt[a]-=1
        for a in range(1, max(seq)+1):
            cnt.setdefault(a,0)
            while cnt[a]<2: out.append(a); cnt[a]+=1
        return out[:2*max(seq)]

    out=[]
    L=len(H)
    base_moves=['swapH','swapV','moveH','moveV','blockH','blockV','revH','revV','pairH','pairV','nestH','nestV','interH','interV','weaveH','weaveV']
    for _ in range(k):
        which = rng.choice(base_moves)
        A = H[:] if 'H' in which else V[:]
        i = rng.randrange(L); j = rng.randrange(L)
        if which.startswith('swap'):
            A[i],A[j]=A[j],A[i]
        elif which.startswith('move'):
            if i!=j:
                x=A.pop(i); A.insert(j,x)
        elif which.startswith('block'):
            a,b=(i,j) if i<j else (j,i)
            if a!=b:
                blk=A[a:b+1]; del A[a:b+1]
                t=rng.randrange(len(A)+1); A[t:t]=blk
        elif which.startswith('rev'):
            a,b=(i,j) if i<j else (j,i)
            if a!=b: A[a:b+1]=list(reversed(A[a:b+1]))
        elif which.startswith('pair'):
            labs=list(set(A))
            if len(labs)>=2:
                a_lab,b_lab=rng.sample(labs,2)
                pa=[p for p,x in enumerate(A) if x==a_lab]
                pb=[p for p,x in enumerate(A) if x==b_lab]
                if len(pa)==2 and len(pb)==2:
                    for ia,ib in zip(pa,pb): A[ia],A[ib]=A[ib],A[ia]
        elif which in ('nestH','nestV'):
            A = apply_nest(A)
        elif which in ('interH','interV'):
            A = apply_interleave(A)
        else:
            A = apply_weave(A)
        out.append(canonicalize(A, V) if 'H' in which else canonicalize(H, A))
    return out

# ---------- MAP-Elites ----------
class MapElites:
    def __init__(self, bd1_bins: int, bd2_bins: int, bd1_range: Tuple[int,int], bd2_range: Tuple[int,int]):
        self.b1_bins = bd1_bins
        self.b2_bins = bd2_bins
        self.r1 = bd1_range
        self.r2 = bd2_range
        self.grid: Dict[Tuple[int,int], Tuple[float, Seq, Seq]] = {}

    def _bin(self, bd1:int, bd2:int) -> Optional[Tuple[int,int]]:
        # map to [0..bins-1]
        if bd1 < self.r1[0] or bd1 > self.r1[1] or bd2 < self.r2[0] or bd2 > self.r2[1]:
            return None
        i = int((bd1 - self.r1[0]) * (self.b1_bins-1) / max(1, self.r1[1]-self.r1[0]))
        j = int((bd2 - self.r2[0]) * (self.b2_bins-1) / max(1, self.r2[1]-self.r2[0]))
        return (i,j)

    def consider(self, ratio: float, H: Seq, V: Seq, bd1:int, bd2:int):
        key = self._bin(bd1, bd2)
        if key is None: return False
        cur = self.grid.get(key)
        if (cur is None) or (ratio > cur[0] + 1e-12):
            self.grid[key] = (ratio, H[:], V[:])
            return True
        return False

    def random_parent(self, rng: random.Random):
        if not self.grid: return None
        key = rng.choice(list(self.grid.keys()))
        return self.grid[key]

    def dump_top(self, k: int):
        items = sorted(self.grid.values(), key=lambda x: -x[0])
        return items[:k]

def load_top_from_pkl(path: str, take: int):
    with open(path,"rb") as f:
        data = pickle.load(f)
    triples=[]
    for it in data:
        try:
            triples.append((float(it[0]), list(it[1]), list(it[2])))
        except Exception:
            pass
    triples.sort(key=lambda x:-x[0])
    return triples[:take]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, default="misr_elites_n14_r*.pkl")
    ap.add_argument("--take", type=int, default=128)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--neighbor_k", type=int, default=220)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--out", type=str, default="qd_archive_n14.pkl")
    # map-elites grid
    ap.add_argument("--bd1_bins", type=int, default=10)
    ap.add_argument("--bd2_bins", type=int, default=10)
    ap.add_argument("--bd1_min", type=int, default=50)
    ap.add_argument("--bd1_max", type=int, default=900)
    ap.add_argument("--bd2_min", type=int, default=2)
    ap.add_argument("--bd2_max", type=int, default=18)
    args = ap.parse_args()

    rng = random.Random(2025)

    # seed archive from input pkl files
    files = sorted(glob.glob(args.glob))
    if not files:
        print("No files matched.")
        return

    arch = MapElites(args.bd1_bins, args.bd2_bins,
                     (args.bd1_min, args.bd1_max),
                     (args.bd2_min, args.bd2_max))

    seeded = 0
    best0 = 0.0
    for f in files:
        triples = load_top_from_pkl(f, args.take)
        for (r,H,V) in triples:
            ratio, bd1, bd2 = score_and_bd(H,V, threads=args.threads)
            arch.consider(ratio, H, V, bd1, bd2)
            best0 = max(best0, ratio); seeded += 1
    print(f"Seeded archive with {seeded} items; best={best0:.4f}; filled={len(arch.grid)} cells")

    # main loop
    best = best0
    t0 = time.time()
    for it in range(1, args.iters+1):
        parent = arch.random_parent(rng)
        if parent is None:
            continue
        _, H0, V0 = parent
        # mutate
        for (H1,V1) in neighbors(H0, V0, rng, k=args.neighbor_k):
            ratio, bd1, bd2 = score_and_bd(H1,V1, threads=args.threads)
            arch.consider(ratio, H1, V1, bd1, bd2)
            if ratio > best + 1e-12:
                best = ratio

        if it % args.save_every == 0:
            tops = arch.dump_top(256)
            with open(args.out, "wb") as f:
                pickle.dump(tops, f)
            print(f"[{it}] filled={len(arch.grid)} cells  best={best:.4f}  saved {len(tops)} -> {args.out}")

    tops = arch.dump_top(256)
    with open(args.out, "wb") as f:
        pickle.dump(tops, f)
    dt = time.time() - t0
    print(f"Done. filled={len(arch.grid)} cells  best={best:.4f}  time={dt:.1f}s  saved {len(tops)} -> {args.out}")

if __name__ == "__main__":
    main()
