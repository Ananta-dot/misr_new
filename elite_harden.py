#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, glob, os, pickle, random, math, time
from typing import List, Tuple, Dict

# ========= minimal shared types / helpers =========
Seq = List[int]
Instance = Tuple[Seq, Seq]
Rect = Tuple[Tuple[int,int], Tuple[int,int]]  # ((x1,x2),(y1,y2))

def canonicalize(H: Seq, V: Seq) -> Instance:
    order = []; seen=set()
    for x in H:
        if x not in seen:
            order.append(x); seen.add(x)
    rel = {old:new for new,old in enumerate(order,1)}
    return [rel[x] for x in H], [rel[x] for x in V]

def instance_key(H: Seq, V: Seq) -> str:
    s = ','.join(map(str, H)) + '|' + ','.join(map(str, V))
    import hashlib
    return hashlib.blake2b(s.encode(), digest_size=16).hexdigest()

def seq_spans(seq: Seq) -> List[Tuple[int, int]]:
    first = {}; spans={}
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

# ========= LP/ILP candidate-point sets =========
def open_midpoints_points(rects: List[Rect]):
    """All consecutive midpoints on each axis + rectangle centers (open interior tests)."""
    xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
    ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
    midx = [ (xs[i]+xs[i+1])/2.0 for i in range(len(xs)-1) ] if len(xs)>1 else []
    midy = [ (ys[i]+ys[i+1])/2.0 for i in range(len(ys)-1) ] if len(ys)>1 else []
    pts = [(x,y) for x in midx for y in midy]
    for (x1,x2),(y1,y2) in rects:  # add centers
        pts.append(((x1+x2)/2.0, (y1+y2)/2.0))
    # dedup
    pts = list({(float(round(x,6)), float(round(y,6))) for (x,y) in pts})
    return pts

def open_all_midpoints_points(rects: List[Rect]):
    """All pairwise midpoints on each axis (not only consecutive) + centers. Bigger set."""
    xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
    ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
    midx = []
    for i in range(len(xs)):
        for j in range(i+1, len(xs)):
            midx.append((xs[i]+xs[j])/2.0)
    midy = []
    for i in range(len(ys)):
        for j in range(i+1, len(ys)):
            midy.append((ys[i]+ys[j])/2.0)
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

# ========= exact LP/ILP with gurobi (on chosen point set) =========
def solve_lp_ilp_from_points(rects: List[Rect], pts, open_interior=True, threads=0):
    import gurobipy as gp
    from gurobipy import GRB
    covers = covers_open(rects, pts) if open_interior else covers_grid_closed(rects, pts)  # closed helper added below

    # LP
    m_lp = gp.Model("misr_lp"); m_lp.setParam('OutputFlag', 0)
    if threads>0: m_lp.setParam('Threads', threads)
    n = len(rects)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m_lp.optimize()
    lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else 0.0

    # ILP
    m_ilp = gp.Model("misr_ilp"); m_ilp.setParam('OutputFlag', 0)
    if threads>0: m_ilp.setParam('Threads', threads)
    y = m_ilp.addVars(n, vtype=GRB.BINARY)
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1)
    m_ilp.optimize()
    ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else 0.0
    return lp, ilp

# (closed helper used only if you switch open_interior=False)
def grid_points(rects: List[Rect]):
    xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
    ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
    return [(x,y) for x in xs for y in ys]

def covers_grid_closed(rects: List[Rect], pts):
    C=[]
    for (x,y) in pts:
        S=[]
        for i,((x1,x2),(y1,y2)) in enumerate(rects):
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                S.append(i)
        C.append(S)
    return C

# unified scorer
def score_ratio(H: Seq, V: Seq, mode: str, threads: int, alpha: float, beta: float):
    rects = build_rects(H,V)
    if mode == "open_midpoints":
        pts = open_midpoints_points(rects)
        lp, ilp = solve_lp_ilp_from_points(rects, pts, open_interior=True, threads=threads)
    elif mode == "open_all_midpoints":
        pts = open_all_midpoints_points(rects)
        lp, ilp = solve_lp_ilp_from_points(rects, pts, open_interior=True, threads=threads)
    elif mode == "closed_grid":
        pts = grid_points(rects)
        lp, ilp = solve_lp_ilp_from_points(rects, pts, open_interior=False, threads=threads)
    else:
        raise ValueError(f"unknown lp_mode {mode}")

    ratio = (lp/ilp) if ilp>0 else 0.0
    n = max(H) if H else 1
    blended = ratio + alpha*(lp/n) - beta*(ilp/n)
    return lp, ilp, ratio, blended

# ========= structure-creating neighbors =========
def apply_nest(seq: Seq, rng: random.Random, k=None) -> Seq:
    labs=list(set(seq))
    if len(labs)<3: return seq[:]
    if k is None: k = rng.randint(3, min(6, len(labs)))
    chosen = rng.sample(labs, k)
    rest = [x for x in seq if x not in chosen]
    block = chosen[:] + list(reversed(chosen))
    pos = rng.randrange(len(rest)+1)
    return rest[:pos] + block + rest[pos:]

def apply_interleave(seq: Seq, rng: random.Random, pairs=None) -> Seq:
    labs=list(set(seq))
    if len(labs)<2: return seq[:]
    if pairs is None: pairs = rng.randint(1, min(4, len(labs)//2))
    chosen = rng.sample(labs, pairs*2)
    rest = [x for x in seq if x not in chosen]
    block=[]
    for i in range(0,len(chosen),2):
        a,b=chosen[i],chosen[i+1]
        block += [a,b,a,b]
    pos = rng.randrange(len(rest)+1)
    return rest[:pos] + block + rest[pos:]

def apply_weave(seq: Seq, rng: random.Random, k=None) -> Seq:
    """Make staircase-like crossing: a1,a2,...,ak, ak,...,a2,a1 with gaps in between."""
    labs=list(set(seq))
    if len(labs)<3: return seq[:]
    if k is None: k = rng.randint(3, min(7, len(labs)))
    chosen = rng.sample(labs, k)
    # place first occurrences left-to-right, second occurrences right-to-left
    pos = list(range(0, 2*len(labs), max(1, 2*len(labs)//(k+1))))
    pos = pos[:k]
    out = seq[:]  # start from existing order but try to reposition chosen labels
    # remove chosen labels
    out = [x for x in out if x not in chosen]
    # insert first occurrences
    for i,a in enumerate(chosen):
        p = min(len(out), pos[i] if i < len(pos) else len(out))
        out.insert(p, a)
    # insert second occurrences reversed
    for i,a in enumerate(reversed(chosen)):
        p = min(len(out), len(out)- (pos[i] if i < len(pos) else 0))
        out.insert(p, a)
    # repair counts to exactly two per label
    cnt = {}
    for x in out: cnt[x] = cnt.get(x,0)+1
    for a in list(cnt.keys()):
        if cnt[a] > 2:
            # remove extras from middle
            need = cnt[a]-2
            i=0
            while need>0 and i < len(out):
                if out[i]==a:
                    del out[i]; need-=1
                else:
                    i+=1
    for a in range(1, max(seq)+1):
        cnt.setdefault(a,0)
        while cnt[a] < 2:
            out.append(a); cnt[a]+=1
    return out[:2*max(seq)]

def neighbors(H: Seq, V: Seq, rng: random.Random, k: int = 240) -> List[Instance]:
    out=[]
    L=len(H)
    moves=['swapH','swapV','moveH','moveV','blockH','blockV','revH','revV','pairH','pairV','nestH','nestV','interH','interV','weaveH','weaveV']
    for _ in range(k):
        which = rng.choice(moves)
        A = H[:] if 'H' in which else V[:]
        i = rng.randrange(L); j = rng.randrange(L)
        if which.startswith('swap'):
            A[i],A[j]=A[j],A[i]
        elif which.startswith('move'):
            if i!=j:
                x=A.pop(i); A.insert(j,x)
        elif which.startswith('block'):
            a,b = (i,j) if i<j else (j,i)
            if a!=b:
                blk=A[a:b+1]; del A[a:b+1]
                t=rng.randrange(len(A)+1); A[t:t]=blk
        elif which.startswith('rev'):
            a,b = (i,j) if i<j else (j,i)
            if a!=b:
                A[a:b+1] = list(reversed(A[a:b+1]))
        elif which.startswith('pair'):
            labs=list(set(A))
            if len(labs)>=2:
                a_lab,b_lab=rng.sample(labs,2)
                pa=[idx for idx,x in enumerate(A) if x==a_lab]
                pb=[idx for idx,x in enumerate(A) if x==b_lab]
                if len(pa)==2 and len(pb)==2:
                    for ia,ib in zip(pa,pb):
                        A[ia],A[ib]=A[ib],A[ia]
        elif which in ('nestH','nestV'):
            A = apply_nest(A, rng)
        elif which in ('interH','interV'):
            A = apply_interleave(A, rng)
        else:
            A = apply_weave(A, rng)
        out.append(canonicalize(A, V) if 'H' in which else canonicalize(H, A))
    return out

# ========= local search with early-blend / late-ratio schedule =========
def local_search(seed: Instance, seconds: float, rng: random.Random,
                 lp_mode: str, threads: int,
                 alpha_hi: float, beta_hi: float,
                 alpha_lo: float, beta_lo: float,
                 phase_split: float = 0.6,
                 tabu_seconds: float=25.0, elite_size: int=64,
                 neighbor_k: int=240,
                 T0: float=0.12, Tmin: float=0.02):
    start=time.time()
    H,V=canonicalize(*seed)
    seen: Dict[str, Tuple[float,float,float,float]]={}
    elites: List[Tuple[float, Seq, Seq]]=[]
    tabu: Dict[str,float]={}
    best=-1.0

    def push(s,h,v):
        elites.append((s,h[:],v[:]))
        elites.sort(key=lambda x:-x[0])
        if len(elites)>elite_size: elites.pop()

    while time.time()-start < seconds:
        frac = (time.time()-start)/max(seconds,1e-6)
        # schedule
        if frac < phase_split:
            alpha, beta = alpha_hi, beta_hi   # exploration: boost LP, lightly penalize ILP
        else:
            alpha, beta = alpha_lo, beta_lo   # exploitation: closer to pure ratio

        key=instance_key(H,V); now=time.time()
        if key in tabu and (now-tabu[key] < tabu_seconds):
            if elites: _,H,V = random.choice(elites)
            else: H,V = H[::-1],V[::-1]
            continue

        if key not in seen:
            lp,ilp,r,b = score_ratio(H,V,lp_mode,threads,alpha,beta)
            seen[key]=(lp,ilp,r,b); push(r,H,V); best=max(best,r)
        else:
            # recompute blended under current alpha/beta for acceptance
            lp,ilp,r,_ = seen[key]
            _,_,_,b = score_ratio(H,V,lp_mode,threads,alpha,beta)

        cand = neighbors(H,V,rng,neighbor_k)
        best_nb=None; best_b=-1e9
        for (h2,v2) in cand:
            k2=instance_key(h2,v2)
            if k2 in seen:
                lp2,ilp2,r2,_ = seen[k2]
                _,_,_,b2 = score_ratio(h2,v2,lp_mode,threads,alpha,beta)
            else:
                lp2,ilp2,r2,b2 = score_ratio(h2,v2,lp_mode,threads,alpha,beta)
                seen[k2]=(lp2,ilp2,r2,b2); push(r2,h2,v2)
            if b2>best_b:
                best_b=b2; best_nb=(h2,v2,lp2,ilp2,r2,b2)

        if best_nb:
            _,_,lp2,ilp2,r2,b2 = best_nb
            T = T0*(1.0-frac) + Tmin
            if b2 >= b:
                H,V = best_nb[0],best_nb[1]
            else:
                if math.exp((b2-b)/max(T,1e-6)) > random.random():
                    H,V = best_nb[0],best_nb[1]
                else:
                    tabu[key]=now
                    if elites: _,H,V = random.choice(elites)
                    else: H,V = H[::-1],V[::-1]
            best=max(best,r2)

    elites.sort(key=lambda x:-x[0])
    return elites, best

# ========= driver: multi-schedule restarts per elite =========
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
    ap.add_argument("--glob", type=str, default="misr_elites_n*_r*.pkl")
    ap.add_argument("--take", type=int, default=64)
    ap.add_argument("--time_per", type=float, default=4.0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--lp_mode", type=str, default="open_all_midpoints",
                    choices=["open_midpoints","open_all_midpoints","closed_grid"])
    # schedules (explore -> exploit)
    ap.add_argument("--alpha_hi", type=float, default=0.55)
    ap.add_argument("--beta_hi",  type=float, default=0.05)
    ap.add_argument("--alpha_lo", type=float, default=0.15)
    ap.add_argument("--beta_lo",  type=float, default=0.30)
    ap.add_argument("--phase_split", type=float, default=0.6)
    ap.add_argument("--restarts", type=int, default=3, help="number of independent schedules per elite")
    ap.add_argument("--neighbor_k", type=int, default=240)
    ap.add_argument("--save_suffix", type=str, default="h2")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print("No files matched.")
        return

    rng_master = random.Random(2025)

    for f in files:
        triples = load_top_from_pkl(f, args.take)
        if not triples:
            continue
        best_before = triples[0][0]
        improved=[]

        for (r0,H0,V0) in triples:
            # try several restart schedules per elite
            for _ in range(args.restarts):
                rng = random.Random(rng_master.randrange(10**9))
                elites, _ = local_search(
                    seed=(H0,V0),
                    seconds=args.time_per,
                    rng=rng,
                    lp_mode=args.lp_mode,
                    threads=args.threads,
                    alpha_hi=args.alpha_hi, beta_hi=args.beta_hi,
                    alpha_lo=args.alpha_lo, beta_lo=args.beta_lo,
                    phase_split=args.phase_split,
                    neighbor_k=args.neighbor_k
                )
                improved.extend(elites)

        if improved:
            improved.sort(key=lambda x:-x[0])
            out_path = f"{os.path.splitext(f)[0]}_{args.save_suffix}.pkl"
            with open(out_path,"wb") as g:
                pickle.dump(improved[:128], g)
            print(f"{os.path.basename(f)}: best_before={best_before:.4f}  best_after={improved[0][0]:.4f}  -> {os.path.basename(out_path)}")
        else:
            print(f"{os.path.basename(f)}: no improvements")

if __name__ == "__main__":
    main()
