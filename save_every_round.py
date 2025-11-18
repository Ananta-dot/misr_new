#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save_every_round.py

PatternBoost-style worst-case MISR instance generator (exact LP/ILP via Gurobi),
with per-round checkpointing and parallel local search.

What’s inside
-------------
- Exact evaluator:
    * Closed rectangles induced by (H,V) spans
    * LP constraints on the FULL grid (all (x_i,y_j)), ILP the same
    * Duals (LP shadow prices) extracted to guide the search
- Global: configurable causal Transformer learns elite patterns and proposes seeds
- Local: tabu + greedy + SA; blended score modes:
    * simple:        ratio + α·(LP/n) − β·(ILP/n)
    * dual:          simple + γ·(dual_gain signal)
    * dual+corners:  dual + λ_overlap·(heavy overlap) + λ_corner·(corner load)
- Curriculum: dual-aware lift n → n + lift_step (inserts pairs at "hot" positions)
- Parallelism: local_search tasks split over processes (--workers) + Gurobi threads
- Apple Silicon friendly: uses torch.mps if available; Gurobi is CPU-bound
- Checkpointing:
    * Every round:   misr_elites_n{n}_r{round}.pkl   (top-256)
    * At each size:  misr_elites_n{n}_final.pkl      (top-256)
    * Very end:      misr_elites.pkl                 (top-256)

Recommended first run on a 14-core CPU Mac:
  --workers 4 --grb_threads 3
"""

from __future__ import annotations
import argparse, math, os, random, sys, time, tempfile, hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ---- torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    print("ERROR: PyTorch is required. Install with `pip install torch`.", file=sys.stderr)
    raise

# Clamp PyTorch CPU threads so it won't compete with Gurobi
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# ---- gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    print("ERROR: gurobipy is required with a valid license.", file=sys.stderr)
    raise

# =========================
# Device selection (MPS friendly)
# =========================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
if hasattr(torch, "set_float32_matmul_precision"):
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# =========================
# Representation & utils
# =========================
Seq = List[int]
Instance = Tuple[Seq, Seq]

SPECIAL = {"BOS": 0, "SEP": 1, "EOS": 2}
BASE_VOCAB = 3
MAX_N = 256  # safety ceiling for embeddings

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

def canonicalize(H: Seq, V: Seq) -> Instance:
    """Relabel by H's first-appearance order to collapse isomorphisms."""
    order = []
    seen = set()
    for x in H:
        if x not in seen:
            order.append(x); seen.add(x)
    rel = {old: new for new, old in enumerate(order, 1)}
    return [rel[x] for x in H], [rel[x] for x in V]

def instance_key(H: Seq, V: Seq) -> str:
    s = ','.join(map(str, H)) + '|' + ','.join(map(str, V))
    return hashlib.blake2b(s.encode(), digest_size=16).hexdigest()

def random_valid_seq(n: int, rng: random.Random) -> Seq:
    """Length 2n with each label twice; uniformly via multiset shuffle."""
    seq = [i for i in range(1, n + 1) for _ in range(2)]
    rng.shuffle(seq)
    return seq

# ===== motifs (deterministic seeds + corner-heavy gadgets) =====
def motif_rainbow(n: int) -> Seq:
    return list(range(1, n+1)) + list(range(n, 0, -1))

def motif_doubled(n: int) -> Seq:
    return [x for i in range(1, n+1) for x in (i,i)]

def motif_interleave(n: int) -> Seq:
    out=[]
    for i in range(1, n+1, 2):
        j = i+1 if i+1<=n else i
        out += [i, j, i, j]
    # ensure exactly two per label
    cnt = {i:0 for i in range(1,n+1)}
    fixed=[]
    for x in out:
        if cnt[x] < 2:
            fixed.append(x); cnt[x]+=1
    for i in range(1,n+1):
        while cnt[i] < 2:
            fixed.append(i); cnt[i]+=1
    return fixed[:2*n]

def motif_zipper(n: int) -> Seq:
    out=[]
    for i in range(1, (n//2)+1):
        j = n - i + 1
        out += [i, j, i, j]
    if n % 2 == 1:
        k = (n//2)+1
        out += [k,k]
    return out[:2*n]

def motif_ladder(n: int) -> Seq:
    out=[]
    a, b = 1, 2
    while len(out) < 2*n:
        out += [a, b if b<=n else a]
        a += 1
        b += 1
        if a > n: a = n
        if b > n: b = n
    # adjust to exactly two per label
    cnt = {i:0 for i in range(1,n+1)}
    fixed=[]
    for x in out:
        if cnt[x] < 2:
            fixed.append(x); cnt[x]+=1
    for i in range(1,n+1):
        while cnt[i] < 2:
            fixed.append(i); cnt[i]+=1
    return fixed[:2*n]

def motif_corner_combo(n: int) -> Tuple[Seq, Seq]:
    return motif_rainbow(n), motif_doubled(n)

def motif_seeds(n: int) -> List[Seq]:
    S = [
        motif_rainbow(n),
        motif_doubled(n),
        motif_interleave(n),
        motif_zipper(n),
        motif_ladder(n),
        list(range(1, n+1)) + list(range(1, n+1)),
        [x for pair in zip(range(1,n+1), range(1,n+1)) for x in pair],  # 1,1,2,2,...
    ]
    out=[]
    for s in S:
        if len(s)==2*n and all(s.count(i)==2 for i in range(1,n+1)):
            out.append(s)
    return out

# =========================
# Exact evaluator with duals + features
# =========================
Rect = Tuple[Tuple[int,int], Tuple[int,int]]  # ((x1,x2),(y1,y2)) with x1<=x2, y1<=y2
Point = Tuple[int,int]

def build_rects(H: Seq, V: Seq) -> List[Rect]:
    X = seq_spans(H); Y = seq_spans(V)
    rects = []
    for (x1,x2),(y1,y2) in zip(X,Y):
        if x1>x2: x1,x2=x2,x1
        if y1>y2: y1,y2=y2,y1
        rects.append(((x1,x2),(y1,y2)))
    return rects

def grid_points(rects: List[Rect]) -> List[Point]:
    xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
    ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
    return [(x,y) for x in xs for y in ys]

def covers_grid_closed(rects: List[Rect], pts: List[Point]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Returns:
        covers: list over constraints p, each is list of rect indices covering p
        rect_to_constr: list over rect i, each is list of constraint indices p covered by rect i
    """
    covers: List[List[int]] = []
    rect_to_constr: List[List[int]] = [[] for _ in rects]
    for p_idx, (x,y) in enumerate(pts):
        S=[]
        for i,((x1,x2),(y1,y2)) in enumerate(rects):
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                S.append(i)
        covers.append(S)
        for i in S:
            rect_to_constr[i].append(p_idx)
    return covers, rect_to_constr

@dataclass
class EvalInfo:
    lp: float
    ilp: float
    ratio: float
    covers: List[List[int]]              # constraints -> rects
    rect_to_constr: List[List[int]]      # rect -> constraints
    duals: List[float]                   # dual π for each constraint
    dual_gain_per_rect: List[float]      # sum π over constraints per rect
    overlap_heavy_count: int             # #points with coverage >= 3
    corner_load: int                     # sum of coincident corners on grid points
    H_heat: List[float]                  # dual heat projected on H indices
    V_heat: List[float]                 # dual heat projected on V indices

# Global cache (per process) to avoid recomputing evals
EVAL_CACHE: Dict[str, EvalInfo] = {}

def _compute_corner_load(rects: List[Rect]) -> int:
    cnt = {}
    for ((x1,x2),(y1,y2)) in rects:
        for pt in [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]:
            cnt[pt] = cnt.get(pt, 0) + 1
    return sum(cnt.values())

def _project_dual_heat_H(rects: List[Rect], dual_gain: List[float], L: int) -> List[float]:
    heat = [0.0]*L
    for i,((x1,x2),_) in enumerate(rects):
        if x2 >= x1:
            w = dual_gain[i] / max(1, (x2 - x1 + 1))
            for x in range(x1, x2+1):
                heat[x] += w
    return heat

def _project_dual_heat_V(rects: List[Rect], dual_gain: List[float], L: int) -> List[float]:
    heat = [0.0]*L
    for i,(_, (y1,y2)) in enumerate(rects):
        if y2 >= y1:
            w = dual_gain[i] / max(1, (y2 - y1 + 1))
            for y in range(y1, y2+1):
                heat[y] += w
    return heat

def solve_lp_ilp_with_duals(rects: List[Rect], grb_threads: int = 0) -> Tuple[float, float, List[float], List[List[int]], List[List[int]]]:
    pts = grid_points(rects)
    covers, rect_to_constr = covers_grid_closed(rects, pts)

    # LP
    m_lp = gp.Model("misr_lp"); m_lp.setParam('OutputFlag', 0)
    if grb_threads > 0: m_lp.setParam('Threads', grb_threads)
    # speed knobs that often help here
    m_lp.setParam("Method", 1)       # dual simplex
    # m_lp.setParam("Presolve", 2)

    n = len(rects)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x')
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    constr_handles = []
    for S in covers:
        if S:
            c = m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1.0)
            constr_handles.append(c)
        else:
            constr_handles.append(None)
    m_lp.optimize()
    lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else 0.0

    duals = [0.0]*len(covers)
    if m_lp.status == GRB.OPTIMAL:
        for idx, ch in enumerate(constr_handles):
            if ch is not None:
                duals[idx] = ch.Pi

    # ILP
    m_ilp = gp.Model("misr_ilp"); m_ilp.setParam('OutputFlag', 0)
    if grb_threads > 0: m_ilp.setParam('Threads', grb_threads)
    m_ilp.setParam("MIPFocus", 1)    # quickly find good incumbents
    m_ilp.setParam("Heuristics", 0.1)
    # m_ilp.setParam("Cuts", 2)

    y = m_ilp.addVars(n, vtype=GRB.BINARY, name='y')
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1.0)
    m_ilp.optimize()
    ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else 0.0

    return lp, ilp, duals, covers, rect_to_constr

def evaluate_instance(H: Seq, V: Seq, grb_threads: int = 0) -> EvalInfo:
    key = instance_key(H,V)
    if key in EVAL_CACHE:
        return EVAL_CACHE[key]

    rects = build_rects(H,V)
    Lh, Lv = len(H), len(V)

    lp, ilp, duals, covers, rect_to_constr = solve_lp_ilp_with_duals(rects, grb_threads=grb_threads)
    ratio = (lp/ilp) if ilp > 0 else 0.0

    # dual gain per rect
    dual_gain = [0.0]*len(rects)
    for i, cons in enumerate(rect_to_constr):
        dual_gain[i] = sum(duals[p] for p in cons)

    # overlap features
    cov_counts = [len(S) for S in covers]
    overlap_heavy_count = sum(1 for c in cov_counts if c >= 3)
    corner_load = _compute_corner_load(rects)

    # project dual to sequence positions (for lift / neighborhood bias)
    H_heat = _project_dual_heat_H(rects, dual_gain, Lh)
    V_heat = _project_dual_heat_V(rects, dual_gain, Lv)

    info = EvalInfo(
        lp=lp, ilp=ilp, ratio=ratio,
        covers=covers, rect_to_constr=rect_to_constr,
        duals=duals, dual_gain_per_rect=dual_gain,
        overlap_heavy_count=overlap_heavy_count,
        corner_load=corner_load,
        H_heat=H_heat, V_heat=V_heat
    )
    EVAL_CACHE[key] = info
    return info

def blended_score(info: EvalInfo, n: int,
                  bias_mode: str,
                  alpha_lp: float, beta_ilp: float,
                  gamma_dual: float,
                  lambda_overlap: float, lambda_corner: float) -> float:
    # base
    ratio = info.ratio
    lp = info.lp; ilp = info.ilp
    sc = ratio + alpha_lp * (lp / max(1,n)) - beta_ilp * (ilp / max(1,n))

    if bias_mode in ("dual", "dual+corners"):
        g = np.array(info.dual_gain_per_rect, dtype=float)
        dual_signal = float(np.mean(g) if g.size > 0 else 0.0)
        sc += gamma_dual * dual_signal

    if bias_mode == "dual+corners":
        sc += lambda_overlap * (info.overlap_heavy_count / max(1,n))
        sc += lambda_corner * (info.corner_load / max(1,n))

    return float(sc)

# =========================
# Local search (tabu + greedy, richer neighbors)
# =========================
def neighbors(H: Seq, V: Seq, rng: random.Random, k: int = 120) -> List[Instance]:
    out=[]
    L = len(H)
    moves = ['swapH','swapV','moveH','moveV','blockH','blockV','revH','revV','pairH','pairV']
    for _ in range(k):
        which = rng.choice(moves)
        A = H[:] if 'H' in which else V[:]
        i = rng.randrange(L); j = rng.randrange(L)
        if which.startswith('swap'):
            A[i], A[j] = A[j], A[i]
        elif which.startswith('move'):
            if i!=j:
                x = A.pop(i); A.insert(j, x)
        elif which.startswith('block'):
            a,b = (i,j) if i<j else (j,i)
            if a!=b:
                blk = A[a:b+1]; del A[a:b+1]
                t = rng.randrange(len(A)+1); A[t:t]=blk
        elif which.startswith('rev'):
            a,b = (i,j) if i<j else (j,i)
            if a!=b:
                A[a:b+1] = list(reversed(A[a:b+1]))
        else:
            # pair-swap entire label pairs
            labs = list(set(A))
            if len(labs) >= 2:
                a_lab, b_lab = rng.sample(labs, 2)
                pa = [idx for idx,x in enumerate(A) if x==a_lab]
                pb = [idx for idx,x in enumerate(A) if x==b_lab]
                if len(pa)==2 and len(pb)==2:
                    for ia, ib in zip(pa, pb):
                        A[ia], A[ib] = A[ib], A[ia]
        out.append(canonicalize(A, V) if 'H' in which else canonicalize(H, A))
    return out

def local_search(seed: Instance,
                 time_budget_s: float,
                 rng: random.Random,
                 bias_mode: str,
                 alpha_lp: float, beta_ilp: float,
                 gamma_dual: float,
                 lambda_overlap: float, lambda_corner: float,
                 grb_threads: int = 0,
                 tabu_seconds: float = 20.0,
                 elite_size: int = 64,
                 neighbor_k: int = 120,
                 sa_T0: float = 0.06):
    """Return (elites_sorted, best_ratio)."""
    start = time.time()
    H, V = canonicalize(*seed)
    n = max(H) if H else 1
    elites: List[Tuple[float, Seq, Seq]] = []  # (ratio, H, V)
    tabu: Dict[str, float] = {}
    best = -1.0

    def push(score: float, h: Seq, v: Seq):
        elites.append((score, h[:], v[:]))
        elites.sort(key=lambda x: -x[0])
        if len(elites) > elite_size: elites.pop()

    # Evaluate starting point
    info = evaluate_instance(H,V, grb_threads=grb_threads)
    blended0 = blended_score(info, n, bias_mode, alpha_lp, beta_ilp, gamma_dual, lambda_overlap, lambda_corner)
    push(info.ratio, H, V)
    best = max(best, info.ratio)

    while time.time() - start < time_budget_s:
        key = instance_key(H,V); now = time.time()
        if key in tabu and (now - tabu[key] < tabu_seconds):
            if elites: _, H, V = random.choice(elites)
            else: H, V = H[::-1], V[::-1]
            info = evaluate_instance(H,V, grb_threads=grb_threads)
            blended0 = blended_score(info, n, bias_mode, alpha_lp, beta_ilp, gamma_dual, lambda_overlap, lambda_corner)
            continue

        cand = neighbors(H,V,rng,neighbor_k)
        best_nb=None; best_sc=-1e18
        for (h2,v2) in cand:
            n2 = max(h2) if h2 else 1
            info2 = evaluate_instance(h2,v2, grb_threads=grb_threads)
            sc2 = blended_score(info2, n2, bias_mode, alpha_lp, beta_ilp, gamma_dual, lambda_overlap, lambda_corner)
            push(info2.ratio, h2, v2)
            if sc2 > best_sc:
                best_sc = sc2; best_nb=(h2,v2,info2,sc2)
        if best_nb:
            h2,v2,info2,sc2 = best_nb
            if sc2 >= blended0:
                H, V = h2, v2
                info = info2
                blended0 = sc2
            else:
                # SA accept by blended score
                delta = sc2 - blended0
                T = sa_T0
                if math.exp(delta/max(T,1e-6)) > random.random():
                    H, V = h2, v2
                    info = info2
                    blended0 = sc2
                else:
                    tabu[key] = now
            best = max(best, info2.ratio)
        else:
            # fallback random shake
            H, V = H[::-1], V[::-1]
            info = evaluate_instance(H,V, grb_threads=grb_threads)
            blended0 = blended_score(info, n, bias_mode, alpha_lp, beta_ilp, gamma_dual, lambda_overlap, lambda_corner)

    elites_sorted = sorted(elites, key=lambda x: -x[0])
    return elites_sorted, best

# =========================
# Transformer proposer (configurable & stronger)
# =========================
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        idx = torch.arange(L, device=x.device)
        return x + self.pe(idx).unsqueeze(0)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]

class BetterGPT(nn.Module):
    """
    Causal TransformerEncoder (decoder-style via causal mask) with n-conditioning.
    Heavier by default than the original TinyGPT, but all dims are CLI-configurable.
    """
    def __init__(self,
                 d_model=384, nhead=8, nlayers=8, dropout=0.10,
                 pos_type="learned", max_len=4096,
                 tie_weights: bool=True):
        super().__init__()
        self.vocab_size = BASE_VOCAB + MAX_N
        self.label_embed = nn.Embedding(self.vocab_size, d_model)
        self.n_embed     = nn.Embedding(MAX_N + 1, d_model)

        if pos_type == "learned":
            self.pos = LearnedPositionalEncoding(max_len=max_len, d_model=d_model)
        else:
            self.pos = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4*d_model, dropout=dropout, batch_first=True,
            activation='gelu'
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.out = nn.Linear(d_model, self.vocab_size, bias=True)

        self.tie_weights = tie_weights
        if self.tie_weights:
            # weight tying improves sample quality & stability
            self.out.weight = self.label_embed.weight

    def forward(self, tokens, n_scalar):
        tok_emb = self.label_embed(tokens)                 # [B, L, d]
        n_emb  = self.n_embed(n_scalar).unsqueeze(1)       # [B, 1, d]
        n_emb  = n_emb.expand(-1, tok_emb.size(1), -1)     # [B, L, d]
        x = self.pos(tok_emb + n_emb)                      # [B, L, d]
        L = x.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
        h = self.enc(x, mask=causal)                       # [B, L, d]
        return self.out(h)                                 # [B, L, vocab]

def seq_to_tokens(seq: Seq) -> List[int]:
    return [BASE_VOCAB + (i-1) for i in seq]

def tokens_to_seq(tokens: List[int]) -> Seq:
    return [t - BASE_VOCAB + 1 for t in tokens]

@dataclass
class Batch:
    tokens: torch.Tensor    # [B, L] (long)
    n_scalar: torch.Tensor  # [B] (long)
    targets: torch.Tensor   # [B, L] (long)

def make_batch(elites: List[Tuple[float, Seq, Seq]], B: int, rng: random.Random) -> Batch:
    tlist=[]; tglist=[]; ns=[]
    for _ in range(B):
        _, H, V = rng.choice(elites)
        n = max(H)
        tok = [SPECIAL["BOS"]] + seq_to_tokens(H) + [SPECIAL["SEP"]] + seq_to_tokens(V) + [SPECIAL["EOS"]]
        tgt = tok[1:] + [SPECIAL["EOS"]]
        tlist.append(torch.tensor(tok, dtype=torch.long))
        tglist.append(torch.tensor(tgt, dtype=torch.long))
        ns.append(n)
    L = max(len(t) for t in tlist)
    pad = SPECIAL["EOS"]
    tokens = torch.full((B, L), pad, dtype=torch.long)
    targets = torch.full((B, L), pad, dtype=torch.long)
    for i,(t,tt) in enumerate(zip(tlist,tglist)):
        tokens[i,:len(t)] = t
        targets[i,:len(tt)] = tt
    return Batch(tokens=tokens.to(DEVICE),
                 n_scalar=torch.tensor(ns, dtype=torch.long, device=DEVICE),
                 targets=targets.to(DEVICE))

@torch.no_grad()
def sample_model(model: BetterGPT, n: int,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 max_len: int = 4096) -> Instance:
    """Generate exactly 2n labels for H, then SEP, then exactly 2n labels for V."""
    model.eval()
    toks = [SPECIAL["BOS"]]

    def step(mask_valid: List[bool]) -> int:
        inp = torch.tensor(toks, dtype=torch.long, device=DEVICE).unsqueeze(0)
        nvec = torch.tensor([n], dtype=torch.long, device=DEVICE)
        logits = model(inp, nvec)[0, -1]  # [V]
        mask = torch.tensor(mask_valid, device=DEVICE)
        logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(logits/temperature, dim=-1)
        sorted_probs, idx = torch.sort(probs, descending=True)
        csum = torch.cumsum(sorted_probs, dim=-1)
        keep = csum <= top_p
        if not torch.any(keep): keep[0] = True
        p = torch.zeros_like(probs).scatter(0, idx[keep], sorted_probs[keep])
        p = p / p.sum()
        return int(torch.multinomial(p, 1).item())

    # build H (2n labels)
    counts = [0]*(n+1)
    while sum(counts) < 2*n:
        mask = [False]*(BASE_VOCAB + MAX_N)
        for i in range(1, n+1):
            if counts[i] < 2:
                mask[BASE_VOCAB + (i-1)] = True
        toks.append(step(mask))
        lab = toks[-1] - BASE_VOCAB + 1
        counts[lab] += 1

    # insert SEP
    toks.append(SPECIAL["SEP"])

    # build V (2n labels)
    counts = [0]*(n+1)
    while sum(counts) < 2*n and len(toks) < max_len:
        mask = [False]*(BASE_VOCAB + MAX_N)
        for i in range(1, n+1):
            if counts[i] < 2:
                mask[BASE_VOCAB + (i-1)] = True
        toks.append(step(mask))
        lab = toks[-1] - BASE_VOCAB + 1
        counts[lab] += 1

    sep_idx = toks.index(SPECIAL["SEP"])
    H_tok = toks[1:sep_idx]
    V_tok = toks[sep_idx+1:]
    H = tokens_to_seq(H_tok); V = tokens_to_seq(V_tok)
    return canonicalize(H,V)

def train_one_step(model: nn.Module, opt: torch.optim.Optimizer, batch: Batch) -> float:
    model.train()
    logits = model(batch.tokens, batch.n_scalar)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch.targets.reshape(-1),
        ignore_index=SPECIAL["EOS"]
    )
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())

# -------------------------
# checkpoint helpers
def _atomic_save_pickle(obj, path: str):
    import pickle
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def save_round_ckpt(elites, n: int, round_idx: int, topk: int = 256):
    path = f"11-10/misr_elites_n{n}_r{round_idx}.pkl"
    _atomic_save_pickle(elites[:topk], path)
    print(f"   saved checkpoint: {path} (top {topk})")

def save_n_final(elites, n: int, topk: int = 256):
    path = f"11-10/misr_elites_n{n}_final.pkl"
    _atomic_save_pickle(elites[:topk], path)
    print(f"   saved n-final checkpoint: {path} (top {topk})")

# =========================
# Seeding utilities
# =========================
def recombine_seeds(elites: List[Tuple[float, Seq, Seq]], k: int, rng: random.Random) -> List[Instance]:
    out=[]
    pool = elites[:max(k*2, 2)]
    if not pool:
        return out
    for _ in range(k):
        _, H1, _ = rng.choice(pool)
        _, _, V2 = rng.choice(pool)
        h, v = canonicalize(H1, V2)
        out.append((h, v))
    return out

def seeded_pool(n: int, rng: random.Random, base_count: int, policy: str) -> List[Instance]:
    """Build first-round seeds for size n according to policy."""
    if policy == "random":
        return [(random_valid_seq(n, rng), random_valid_seq(n, rng)) for _ in range(base_count)]
    motifs = motif_seeds(n)
    seeds: List[Instance] = []
    # motif-vs-random & random-vs-motif
    for m in motifs:
        seeds.append((m[:], random_valid_seq(n, rng)))
        seeds.append((random_valid_seq(n, rng), m[:]))
    # explicit corner pairs
    RH, RV = motif_corner_combo(n)
    seeds.append((RH[:], RV[:]))
    seeds.append((RV[:], RH[:]))
    # a few pure motif-motif
    for i in range(min(len(motifs)-1, 5 if policy=="mixed" else 7)):
        seeds.append((motifs[i][:], motifs[i+1][:]))
    # fill to count:
    while len(seeds) < base_count:
        if policy == "motif":
            a = rng.choice(motifs); b = rng.choice(motifs)
            seeds.append((a[:], b[:]))
        else:
            seeds.append((random_valid_seq(n, rng), random_valid_seq(n, rng)))
    return seeds[:base_count]

# =========================
# Dual-aware lifting
# =========================
def _choose_index_by_weight(weights: List[float], rng: random.Random) -> int:
    if not weights:
        return 0
    w = np.array(weights, dtype=float)
    w = w - w.min() + 1e-8
    s = float(w.sum())
    if s <= 0:
        return rng.randrange(len(weights))
    r = rng.random() * s
    c = 0.0
    for i, wi in enumerate(w):
        c += float(wi)
        if r <= c:
            return i
    return len(weights)-1

def lift_instance_dualaware(H: Seq, V: Seq, n_new: int, rng: random.Random,
                            H_heat: Optional[List[float]]=None,
                            V_heat: Optional[List[float]]=None) -> Instance:
    """Insert pairs of new labels; if heat maps are provided, use them."""
    assert n_new >= max(H)
    H2, V2 = H[:], V[:]
    for lab in range(max(H)+1, n_new+1):
        for (seq, heat) in ((H2, H_heat), (V2, V_heat)):
            if heat is None or len(heat) != len(seq):
                # fallback to simple depth-based
                sp = seq_spans(seq)
                line = [0]*(len(seq)+1)
                for (l,r) in sp:
                    if l < r:
                        line[l] += 1
                        if r+1 < len(line):
                            line[r+1] -= 1
                d=[]; cur=0
                for i in range(len(seq)):
                    cur += line[i]; d.append(cur)
                i = _choose_index_by_weight(d, rng)
                j = min(i+1, len(seq))
            else:
                i = _choose_index_by_weight(heat, rng)
                j = min(i+1, len(seq))
            seq.insert(i, lab); seq.insert(j, lab)
    return canonicalize(H2, V2)

# =========================
# Parallel worker wrapper
# =========================
def _search_worker(args_tuple):
    (seed_pair, rng_seed, time_budget_s, bias_mode, alpha_lp, beta_ilp,
     gamma_dual, lambda_overlap, lambda_corner, grb_threads) = args_tuple
    rng = random.Random(rng_seed)
    return local_search(
        seed_pair,
        time_budget_s=time_budget_s,
        rng=rng,
        bias_mode=bias_mode,
        alpha_lp=alpha_lp, beta_ilp=beta_ilp,
        gamma_dual=gamma_dual,
        lambda_overlap=lambda_overlap, lambda_corner=lambda_corner,
        grb_threads=grb_threads,
        elite_size=64,
        neighbor_k=120,
        sa_T0=0.06,
    )

# =========================
# PatternBoost Runner
# =========================
def run_patternboost(
    seed: int = 123,
    n_start: int = 8,
    n_target: int = 32,
    rounds_per_n: int = 10,
    seeds_per_round: int = 32,
    local_time_per_seed: float = 3.0,
    elites_to_train: int = 96,
    batch_size: int = 32,
    train_steps_per_round: int = 60,
    temperature: float = 1.0,
    top_p: float = 0.9,
    # search shaping
    bias_mode: str = "dual+corners",  # simple | dual | dual+corners
    alpha_lp: float = 0.15,
    beta_ilp: float = 0.10,
    gamma_dual: float = 0.15,
    lambda_overlap: float = 0.10,
    lambda_corner: float = 0.05,
    grb_threads: int = 0,
    lift_step: int = 3,
    seed_policy: str = "mixed",  # mixed | random | motif
    # transformer cfg
    d_model: int = 384,
    nhead: int = 8,
    nlayers: int = 8,
    dropout: float = 0.10,
    pos_type: str = "learned",
    tie_weights: bool = True,
    # parallelism
    workers: int = 1,
):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    model = BetterGPT(d_model, nhead, nlayers, dropout, pos_type, 4096, tie_weights).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    elites: List[Tuple[float, Seq, Seq]] = []
    def push_elite(score, H, V):
        elites.append((score, H[:], V[:]))
        elites.sort(key=lambda x: -x[0])
        if len(elites) > 4096: elites[:] = elites[:4096]

    n = n_start
    best_overall = 0.0
    seeds = seeded_pool(n, rng, seeds_per_round, policy=seed_policy)

    while n <= n_target:
        print(f"\n=== SIZE n={n} ({len(seeds)} seeds) ===")
        for r in range(rounds_per_n):
            # 1) Local search (parallel or single)
            if workers <= 1:
                for (H, V) in seeds:
                    es, best = local_search(
                        (H,V),
                        time_budget_s=local_time_per_seed,
                        rng=rng,
                        bias_mode=bias_mode,
                        alpha_lp=alpha_lp, beta_ilp=beta_ilp,
                        gamma_dual=gamma_dual,
                        lambda_overlap=lambda_overlap, lambda_corner=lambda_corner,
                        grb_threads=grb_threads,
                        elite_size=64,
                        neighbor_k=120,
                        sa_T0=0.06
                    )
                    for (score, h, v) in es:
                        push_elite(score, h, v)
                    if best is not None:
                        best_overall = max(best_overall, best)
            else:
                jobs = []
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    for i, (H, V) in enumerate(seeds):
                        jobs.append(ex.submit(
                            _search_worker,
                            ((H, V),
                             rng.randint(0, 10**9),
                             local_time_per_seed,
                             bias_mode, alpha_lp, beta_ilp,
                             gamma_dual, lambda_overlap, lambda_corner,
                             grb_threads)
                        ))
                    for fut in as_completed(jobs):
                        es, best = fut.result()
                        for (score, h, v) in es:
                            push_elite(score, h, v)
                        if best is not None:
                            best_overall = max(best_overall, best)

            print(f"[round {r+1}/{rounds_per_n}] elites={len(elites)} best_so_far={best_overall:.4f}")

            # 2) Train transformer on elites
            topk = elites[:max(elites_to_train, min(32, len(elites)))]
            if topk:
                last_loss = None
                for _ in range(train_steps_per_round):
                    batch = make_batch(topk, min(batch_size, len(topk)), rng)
                    last_loss = train_one_step(model, opt, batch)
                print(f"   trained {train_steps_per_round} steps, last loss ~ {last_loss:.3f}")

            # 3) Save checkpoint after every round
            save_round_ckpt(elites, n=n, round_idx=r+1, topk=256)

            # 4) New seeds: recombine + transformer + motif refresh
            new_seeds: List[Instance] = []
            new_seeds.extend(recombine_seeds(elites[:128], k=max(1, seeds_per_round//4), rng=rng))
            while len(new_seeds) < seeds_per_round:
                elite_mut = (rng.random() < 0.25) and len(elites) > 0
                if elite_mut:
                    _, h, v = rng.choice(elites[:64])
                    h = h[:]; v = v[:]
                    for S in (h, v):
                        if rng.random() < 0.6:
                            i, j = rng.randrange(len(S)), rng.randrange(len(S))
                            S[i], S[j] = S[j], S[i]
                    new_seeds.append((h, v))
                else:
                    h, v = sample_model(model, n, temperature=temperature, top_p=top_p)
                    if (not h) or (not v) or len(h)!=2*n or len(v)!=2*n:
                        h, v = random_valid_seq(n, rng), random_valid_seq(n, rng)
                    # occasional strong jitter (reverse block)
                    if rng.random() < 0.35:
                        for S in (h, v):
                            a = rng.randrange(len(S)); b = rng.randrange(len(S))
                            a, b = min(a,b), max(a,b)
                            if a != b:
                                S[a:b+1] = list(reversed(S[a:b+1]))
                    new_seeds.append((h, v))

            if seed_policy != "random":
                mix = []
                RH, RV = motif_corner_combo(n)
                mix.append((RH[:], RV[:]))
                mix.append((RV[:], RH[:]))
                motifs = motif_seeds(n)[:2]
                for m in motifs:
                    mix.append((m[:], motif_rainbow(n)))
                R = min(len(mix), max(2, seeds_per_round//8))
                for i in range(R):
                    new_seeds[i] = mix[i]

            seeds = new_seeds

        # Save final for this n
        save_n_final(elites, n=n, topk=256)

        # 5) Curriculum lift by +lift_step (dual-aware if info cached)
        if elites:
            n_next = n + lift_step
            lifted=[]
            top_for_lift = elites[:min(96, len(elites))]
            for _, h, v in top_for_lift:
                inf = EVAL_CACHE.get(instance_key(h,v))
                if inf is not None:
                    h2, v2 = lift_instance_dualaware(h, v, n_next, rng, H_heat=inf.H_heat, V_heat=inf.V_heat)
                else:
                    h2, v2 = lift_instance_dualaware(h, v, n_next, rng, H_heat=None, V_heat=None)
                lifted.append((h2,v2))
            seeds = lifted + seeds[:max(0, seeds_per_round - len(lifted))]
            n = n_next
        else:
            seeds = seeded_pool(n, rng, seeds_per_round, policy=seed_policy)

    # Print best few
    print("\n=== BEST ELITES ===")
    for i,(score,h,v) in enumerate(elites[:10]):
        print(f"#{i+1} ratio={score:.4f}  n={max(h)}")
    return elites

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n_start", type=int, default=8)
    ap.add_argument("--n_target", type=int, default=32)
    ap.add_argument("--rounds_per_n", type=int, default=10)
    ap.add_argument("--seeds_per_round", type=int, default=32)
    ap.add_argument("--local_time_per_seed", type=float, default=3.0)
    ap.add_argument("--elites_to_train", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--train_steps_per_round", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--bias_mode", type=str, default="dual+corners",
                    choices=["simple","dual","dual+corners"])
    ap.add_argument("--alpha_lp", type=float, default=0.15)
    ap.add_argument("--beta_ilp", type=float, default=0.10)
    ap.add_argument("--gamma_dual", type=float, default=0.15)
    ap.add_argument("--lambda_overlap", type=float, default=0.10)
    ap.add_argument("--lambda_corner", type=float, default=0.05)

    ap.add_argument("--grb_threads", type=int, default=0)
    ap.add_argument("--lift_step", type=int, default=3)
    ap.add_argument("--seed_policy", type=str, default="mixed",
                    choices=["mixed","random","motif"])

    # Transformer knobs
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--pos_type", type=str, default="learned", choices=["learned","sinusoid"])
    ap.add_argument("--tie_weights", action="store_true", default=True)
    ap.add_argument("--no_tie_weights", dest="tie_weights", action="store_false")

    # Parallelism
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel local_search workers (processes)")

    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    elites: List[Tuple[float, Seq, Seq]] = []
    try:
        elites = run_patternboost(
            seed=args.seed,
            n_start=5,
            n_target=args.n_target,
            rounds_per_n=args.rounds_per_n,
            seeds_per_round=args.seeds_per_round,
            local_time_per_seed=args.local_time_per_seed,
            elites_to_train=args.elites_to_train,
            batch_size=args.batch_size,
            train_steps_per_round=args.train_steps_per_round,
            temperature=args.temperature,
            top_p=args.top_p,
            bias_mode=args.bias_mode,
            alpha_lp=args.alpha_lp,
            beta_ilp=args.beta_ilp,
            gamma_dual=args.gamma_dual,
            lambda_overlap=args.lambda_overlap,
            lambda_corner=args.lambda_corner,
            grb_threads=args.grb_threads,
            lift_step=args.lift_step,
            seed_policy=args.seed_policy,
            d_model=args.d_model,
            nhead=args.nhead,
            nlayers=args.nlayers,
            dropout=args.dropout,
            pos_type=args.pos_type,
            tie_weights=args.tie_weights,
            workers=args.workers,
        )
    finally:
        # Always write a final dump of current elites (best-effort)
        try:
            _atomic_save_pickle(elites[:256], "11-10/misr_elites_11-10.pkl")
            print("Saved top elites to misr_elites.pkl")
        except Exception as e:
            print(f"WARNING: failed to save final elites: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
