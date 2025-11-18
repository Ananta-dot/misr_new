#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatternBoost-style worst-case MISR instance generator (exact LP/ILP via Gurobi).

What's in here:
- Evaluator: two modes
    * open_midpoints  : OPEN rectangles (strict interior) via x/y midpoints + centers
    * closed_grid     : CLOSED rectangles (≤) via full X×Y grid of boundaries
- Global:    Tiny causal Transformer learns elite patterns, proposes new seeds
- Local:     Tabu/greedy + simulated annealing neighborhood search
             + structure-building moves: nest/interleave blocks
- Curriculum: lift by +k (default +3)
- Warm start: seed from previous elites .pkl (lift or shrink to n_start)
- Two-phase schedule per size: inflate LP first, then harden gap
- Checkpoints: per-round pickle save: misr_elites_n{n}_r{round}.pkl and per-size final

Apple Silicon friendly: uses torch.mps if available; Gurobi runs on CPU.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ---- torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    print("ERROR: PyTorch is required. Install with `pip install torch`.", file=sys.stderr)
    raise

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
MAX_N = 128  # safety ceiling

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
    import hashlib
    return hashlib.blake2b(s.encode(), digest_size=16).hexdigest()

def random_valid_seq(n: int, rng: random.Random) -> Seq:
    """Length 2n with each label twice; uniformly via multiset shuffle."""
    seq = [i for i in range(1, n + 1) for _ in range(2)]
    rng.shuffle(seq)
    return seq

def shrink_instance(H: Seq, V: Seq, n_target: int) -> Instance:
    """Remove labels > n_target; keep order of remaining; then canonicalize."""
    keep = set(range(1, n_target+1))
    H2 = [x for x in H if x in keep]
    V2 = [x for x in V if x in keep]
    def fix(seq: Seq):
        cnt = {}
        for x in seq:
            cnt[x] = cnt.get(x, 0) + 1
        bad = {x for x,c in cnt.items() if c!=2}
        return [x for x in seq if x not in bad]
    H2 = fix(H2); V2 = fix(V2)
    return canonicalize(H2, V2)

def lift_instance(H: Seq, V: Seq, n_new: int, rng: random.Random) -> Instance:
    """Insert pairs of new labels near congested areas (curriculum)."""
    assert n_new >= max(H)
    H2, V2 = H[:], V[:]

    def depth(seq: Seq) -> List[int]:
        sp = seq_spans(seq)
        line = [0]*(len(seq)+1)
        for (l,r) in sp:
            if l < r:
                line[l] += 1
                if r+1 < len(line):
                    line[r+1] -= 1
        out=[]; cur=0
        for i in range(len(seq)):
            cur += line[i]; out.append(cur)
        return out

    def weighted_idx(weights: List[int]) -> int:
        tot = sum(w+1 for w in weights)
        x = rng.randrange(tot); s=0
        for i,w in enumerate(weights):
            s += (w+1)
            if x < s: return i
        return len(weights)-1

    for lab in range(max(H)+1, n_new+1):
        for seq in (H2, V2):
            d = depth(seq)
            i = weighted_idx(d); j = min(i+1, len(seq))
            seq.insert(i, lab); seq.insert(j, lab)
    return canonicalize(H2, V2)

# ===== motifs (deterministic seeds + corner-heavy gadgets) =====
def motif_rainbow(n: int) -> Seq:
    return list(range(1, n+1)) + list(range(n, 0, -1))

def motif_doubled(n: int) -> Seq:
    out=[];  [out.extend([i,i]) for i in range(1, n+1)]
    return out

def motif_interleave(n: int) -> Seq:
    out=[]
    for i in range(1, n+1, 2):
        j = i+1 if i+1<=n else i
        out += [i, j, i, j]
    cnt = {i:0 for i in range(1,n+1)}; fixed=[]
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
        a += 1; b += 1
        if a > n: a = n
        if b > n: b = n
    cnt = {i:0 for i in range(1,n+1)}; fixed=[]
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
        [x for pair in zip(range(1,n+1), range(1,n+1)) for x in pair],
    ]
    out=[]
    for s in S:
        if len(s)==2*n and all(s.count(i)==2 for i in range(1,n+1)):
            out.append(s)
    return out

# =========================
# Exact evaluator: two modes
# =========================
Rect = Tuple[Tuple[int,int], Tuple[int,int]]  # ((x1,x2),(y1,y2))

def build_rects(H: Seq, V: Seq) -> List[Rect]:
    X = seq_spans(H); Y = seq_spans(V)
    rects = []
    for (x1,x2),(y1,y2) in zip(X,Y):
        if x1>x2: x1,x2=x2,x1
        if y1>y2: y1,y2=y2,y1
        rects.append(((x1,x2),(y1,y2)))
    return rects

# --- mode: closed_grid (<= on corners)
def grid_points(rects: List[Rect]) -> List[Tuple[int,int]]:
    xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
    ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
    return [(x,y) for x in xs for y in ys]

def covers_grid_closed(rects: List[Rect], pts: List[Tuple[int,int]]) -> List[List[int]]:
    C=[]
    for (x,y) in pts:
        S=[]
        for i,((x1,x2),(y1,y2)) in enumerate(rects):
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                S.append(i)
        C.append(S)
    return C

# --- mode: open_midpoints (< on interior)
def critical_points(rects: List[Rect]) -> List[Tuple[float,float]]:
    xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
    ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
    xs_mid = [ (xs[i]+xs[i+1])/2.0 for i in range(len(xs)-1) ] if len(xs)>1 else []
    ys_mid = [ (ys[i]+ys[i+1])/2.0 for i in range(len(ys)-1) ] if len(ys)>1 else []
    pts = [(x,y) for x in xs_mid for y in ys_mid]
    for (x1,x2),(y1,y2) in rects:
        pts.append(((x1+x2)/2.0, (y1+y2)/2.0))
    pts = list({(float(round(x,6)), float(round(y,6))) for (x,y) in pts})
    return pts

def covers_open(rects: List[Rect], pts: List[Tuple[float,float]]) -> List[List[int]]:
    C=[]
    for (x,y) in pts:
        S=[]
        for i,((x1,x2),(y1,y2)) in enumerate(rects):
            if (x1 < x < x2) and (y1 < y < y2):
                S.append(i)
        C.append(S)
    return C

def solve_lp_ilp(rects: List[Rect], lp_mode: str = "open_midpoints", grb_threads: int = 0) -> Tuple[float, float]:
    assert lp_mode in ("open_midpoints", "closed_grid")
    if lp_mode == "closed_grid":
        pts = grid_points(rects)
        covers = covers_grid_closed(rects, pts)
    else:
        pts = critical_points(rects)
        covers = covers_open(rects, pts)

    # LP
    m_lp = gp.Model("misr_lp"); m_lp.setParam('OutputFlag', 0)
    if grb_threads > 0: m_lp.setParam('Threads', grb_threads)
    n = len(rects)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x')
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m_lp.optimize()
    lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else 0.0

    # ILP
    m_ilp = gp.Model("misr_ilp"); m_ilp.setParam('OutputFlag', 0)
    if grb_threads > 0: m_ilp.setParam('Threads', grb_threads)
    y = m_ilp.addVars(n, vtype=GRB.BINARY, name='y')
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1)
    m_ilp.optimize()
    ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else 0.0

    return lp, ilp

def score_ratio(H: Seq, V: Seq,
                alpha_lp: float = 0.0,
                beta_ilp: float = 0.0,
                lp_mode: str = "open_midpoints",
                grb_threads: int = 0) -> Tuple[float,float,float,float]:
    """Returns (lp, ilp, ratio, blended) where blended guides local search."""
    rects = build_rects(H,V)
    lp, ilp = solve_lp_ilp(rects, lp_mode=lp_mode, grb_threads=grb_threads)
    ratio = (lp/ilp) if ilp > 0 else 0.0
    n = max(H) if H else 1
    blended = ratio + alpha_lp * (lp / n) - beta_ilp * (ilp / n)
    return lp, ilp, ratio, blended

# =========================
# Local search (tabu + greedy + annealing + structure moves)
# =========================
def apply_nest(seq: Seq, rng: random.Random, k: int | None = None) -> Seq:
    labs = list(set(seq))
    if len(labs) < 3: return seq[:]
    if k is None: k = rng.randint(3, min(6, len(labs)))
    chosen = rng.sample(labs, k)
    rest = [x for x in seq if x not in chosen]
    block = chosen[:] + list(reversed(chosen))
    pos = rng.randrange(len(rest)+1)
    return rest[:pos] + block + rest[pos:]

def apply_interleave(seq: Seq, rng: random.Random, pairs: int | None = None) -> Seq:
    labs = list(set(seq))
    if len(labs) < 2: return seq[:]
    if pairs is None: pairs = rng.randint(1, min(4, len(labs)//2))
    chosen = rng.sample(labs, pairs*2)
    rest = [x for x in seq if x not in chosen]
    block=[]
    for i in range(0, len(chosen), 2):
        a, b = chosen[i], chosen[i+1]
        block += [a,b,a,b]
    pos = rng.randrange(len(rest)+1)
    return rest[:pos] + block + rest[pos:]

def neighbors(H: Seq, V: Seq, rng: random.Random, k: int = 160) -> List[Instance]:
    out=[]
    L = len(H)
    moves = ['swapH','swapV','moveH','moveV','blockH','blockV','revH','revV',
             'pairH','pairV','nestH','nestV','interH','interV']
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
        elif which.startswith('pair'):
            labs = list(set(A))
            if len(labs) >= 2:
                a_lab, b_lab = rng.sample(labs, 2)
                pa = [idx for idx,x in enumerate(A) if x==a_lab]
                pb = [idx for idx,x in enumerate(A) if x==b_lab]
                if len(pa)==2 and len(pb)==2:
                    for ia, ib in zip(pa, pb):
                        A[ia], A[ib] = A[ib], A[ia]
        elif which == 'nestH' or which == 'nestV':
            A = apply_nest(A, rng)
        else:  # interH / interV
            A = apply_interleave(A, rng)
        out.append(canonicalize(A, V) if 'H' in which else canonicalize(H, A))
    return out

def local_search(seed: Instance,
                 time_budget_s: float,
                 rng: random.Random,
                 alpha_lp: float,
                 beta_ilp: float,
                 lp_mode: str,
                 grb_threads: int = 0,
                 tabu_seconds: float = 25.0,
                 elite_size: int = 64,
                 neighbor_k: int = 160,
                 T0: float = 0.09,
                 Tmin: float = 0.01):
    """Return (elites_sorted, best_ratio)."""
    start = time.time()
    H, V = canonicalize(*seed)
    seen: Dict[str, Tuple[float,float,float,float]] = {}  # key -> (lp,ilp,ratio,blended)
    elites: List[Tuple[float, Seq, Seq]] = []  # (ratio, H, V)
    tabu: Dict[str, float] = {}
    best = -1.0

    def push(score: float, h: Seq, v: Seq):
        elites.append((score, h[:], v[:]))
        elites.sort(key=lambda x: -x[0])
        if len(elites) > elite_size: elites.pop()

    while time.time() - start < time_budget_s:
        key = instance_key(H,V); now = time.time()
        if key in tabu and (now - tabu[key] < tabu_seconds):
            if elites: _, H, V = random.choice(elites)
            else: H, V = H[::-1], V[::-1]
            continue

        if key not in seen:
            lp, ilp, ratio, blended = score_ratio(H,V,
                                                  alpha_lp=alpha_lp,
                                                  beta_ilp=beta_ilp,
                                                  lp_mode=lp_mode,
                                                  grb_threads=grb_threads)
            seen[key] = (lp, ilp, ratio, blended)
            push(ratio, H, V)
            best = max(best, ratio)
        else:
            lp, ilp, ratio, blended = seen[key]

        cand = neighbors(H,V,rng,neighbor_k)
        best_nb=None; best_sc=-1e9
        for (h2,v2) in cand:
            k2 = instance_key(h2,v2)
            if k2 in seen:
                lp2, ilp2, r2, b2 = seen[k2]
            else:
                lp2, ilp2, r2, b2 = score_ratio(h2,v2,
                                                alpha_lp=alpha_lp,
                                                beta_ilp=beta_ilp,
                                                lp_mode=lp_mode,
                                                grb_threads=grb_threads)
                seen[k2] = (lp2, ilp2, r2, b2)
                push(r2, h2, v2)
            if b2 > best_sc:
                best_sc = b2; best_nb=(h2,v2,lp2,ilp2,r2,b2)

        if best_nb:
            _,_,lp2,ilp2,r2,b2 = best_nb
            # annealing schedule
            elapsed = time.time() - start
            frac = max(0.0, min(1.0, elapsed / max(time_budget_s, 1e-6)))
            T = T0*(1.0-frac) + Tmin
            if b2 >= blended:
                H, V = best_nb[0], best_nb[1]
            else:
                delta = b2 - blended
                if math.exp(delta / max(T,1e-6)) > random.random():
                    H, V = best_nb[0], best_nb[1]
                else:
                    tabu[key] = now
                    if elites: _, H, V = random.choice(elites)
                    else: H, V = H[::-1], V[::-1]
            best = max(best, r2)

    elites_sorted = sorted(elites, key=lambda x: -x[0])
    return elites_sorted, best

# =========================
# Tiny Transformer proposer (encoder-only, causal)
# =========================
class PositionalEncoding(nn.Module):
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

class TinyGPT(nn.Module):
    """Causal TransformerEncoder (decoder-only GPT style) with n-conditioning."""
    def __init__(self, d=192, nhead=6, nlayers=3, dropout=0.1):
        super().__init__()
        self.label_embed = nn.Embedding(BASE_VOCAB + MAX_N, d)
        self.n_embed     = nn.Embedding(MAX_N + 1, d)
        self.pos         = PositionalEncoding(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=4*d,
            dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.out = nn.Linear(d, BASE_VOCAB + MAX_N)

    def forward(self, tokens, n_scalar):
        tok_emb = self.label_embed(tokens)                # [B, L, d]
        n_emb  = self.n_embed(n_scalar).unsqueeze(1)      # [B, 1, d]
        n_emb  = n_emb.expand(-1, tok_emb.size(1), -1)    # [B, L, d]
        x = self.pos(tok_emb + n_emb)                     # [B, L, d]
        L = x.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
        h = self.enc(x, mask=causal)                      # [B, L, d]
        return self.out(h)                                # [B, L, vocab]

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
def sample_model(model: TinyGPT, n: int,
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

def train_one_step(model: nn.Module, opt: torch.optim.Optimizer, batch: Batch):
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

# =========================
# Seeding helpers
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

def seeded_pool(n: int, rng: random.Random, base_count: int) -> List[Instance]:
    """Motif injections + random pairs; includes corner-heavy combos."""
    seeds = []
    motifs = motif_seeds(n)

    # mix motifs as H/V
    for m in motifs:
        seeds.append((m[:], random_valid_seq(n, rng)))
        seeds.append((random_valid_seq(n, rng), m[:]))

    # corner-focused pairs
    RH, RV = motif_corner_combo(n)
    seeds.append((RH[:], RV[:]))
    seeds.append((RV[:], RH[:]))

    # a couple of pure motif pairs
    for i in range(min(len(motifs)-1, 3)):
        seeds.append((motifs[i][:], motifs[i+1][:]))

    # fill with random pairs
    while len(seeds) < base_count:
        seeds.append((random_valid_seq(n, rng), random_valid_seq(n, rng)))

    return seeds[:base_count]

def seeds_from_pickle(path: str, n_start: int, take: int, rng: random.Random) -> List[Instance]:
    if (not path) or (not os.path.exists(path)):
        return []
    import pickle
    with open(path, "rb") as f:
        elites = pickle.load(f)
    triples=[]
    for item in elites:
        try:
            ratio, H, V = float(item[0]), list(item[1]), list(item[2])
            triples.append((ratio, H, V))
        except Exception:
            continue
    triples.sort(key=lambda x: -x[0])
    triples = triples[:take]
    out=[]
    for ratio, H, V in triples:
        n0 = max(H) if H else 0
        if n0 < n_start:
            H2, V2 = lift_instance(H, V, n_start, rng)
        elif n0 > n_start:
            H2, V2 = shrink_instance(H, V, n_start)
        else:
            H2, V2 = canonicalize(H, V)
        out.append((H2, V2))
    seen=set(); uniq=[]
    for h,v in out:
        k = instance_key(h,v)
        if k not in seen:
            seen.add(k); uniq.append((h,v))
    return uniq

# =========================
# PatternBoost Runner
# =========================
def run_patternboost(
    # general
    seed: int = 123,
    n_start: int = 14,
    n_target: int = 32,
    rounds_per_n: int = 14,
    seeds_per_round: int = 128,
    local_time_per_seed: float = 6.0,
    elites_to_train: int = 128,
    batch_size: int = 32,
    train_steps_per_round: int = 60,
    temperature: float = 1.0,
    top_p: float = 0.9,
    # scoring / search
    lp_mode: str = "open_midpoints",  # or "closed_grid"
    # two-phase schedule
    phase_split: float = 0.6,  # fraction of rounds in Phase A
    alpha_lp_hi: float = 0.55, beta_ilp_hi: float = 0.05,  # Phase A (LP inflation)
    alpha_lp_lo: float = 0.20, beta_ilp_lo: float = 0.28,  # Phase B (gap hardening)
    grb_threads: int = 0,
    lift_step: int = 3,
    # warm start
    init_pkl: str = "",
    init_take: int = 192,
    # saving
    save_each_round: bool = True,
):

    rng = random.Random(seed)
    torch.manual_seed(seed)

    model = TinyGPT(d=192, nhead=6, nlayers=3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    elites: List[Tuple[float, Seq, Seq]] = []
    def push_elite(score, H, V):
        elites.append((score, H[:], V[:]))
        elites.sort(key=lambda x: -x[0])
        if len(elites) > 4096: elites[:] = elites[:4096]

    seeds = seeds_from_pickle(init_pkl, n_start, init_take, rng)
    if len(seeds) < seeds_per_round:
        seeds += seeded_pool(n_start, rng, seeds_per_round - len(seeds))
    seeds = seeds[:seeds_per_round]

    n = n_start
    best_overall = 0.0

    while n <= n_target:
        print(f"\n=== SIZE n={n} ({len(seeds)} seeds) ===")
        for r in range(rounds_per_n):
            # phase schedule
            frac = (r+1)/max(rounds_per_n,1)
            if frac <= phase_split:
                alpha_lp, beta_ilp = alpha_lp_hi, beta_ilp_hi
            else:
                alpha_lp, beta_ilp = alpha_lp_lo, beta_ilp_lo

            # 1) Local search
            for (H,V) in seeds:
                es, best = local_search(
                    (H,V),
                    time_budget_s=local_time_per_seed,
                    rng=rng,
                    alpha_lp=alpha_lp,
                    beta_ilp=beta_ilp,
                    lp_mode=lp_mode,
                    grb_threads=grb_threads,
                    elite_size=64,
                    neighbor_k=160,
                    T0=0.09, Tmin=0.01
                )
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

            # 3) New seeds: recombine + transformer + motifs + jitter
            new_seeds = []
            new_seeds.extend(recombine_seeds(elites[:128], k=max(1, seeds_per_round//4), rng=rng))
            while len(new_seeds) < seeds_per_round:
                if elites and rng.random() < 0.25:
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
                    # stronger structure nudges occasionally
                    if rng.random() < 0.35:
                        h = apply_nest(h, rng) if rng.random()<0.5 else apply_interleave(h, rng)
                    if rng.random() < 0.35:
                        v = apply_nest(v, rng) if rng.random()<0.5 else apply_interleave(v, rng)
                    new_seeds.append((h, v))
            # inject a small motif refresh (corner-ish)
            mix = []
            RH, RV = motif_corner_combo(n)
            mix.append((RH[:], RV[:])); mix.append((RV[:], RH[:]))
            motifs = motif_seeds(n)[:2]
            for m in motifs:
                mix.append((m[:], motif_rainbow(n)))
            for i in range(min(len(mix), max(2, seeds_per_round//8))):
                new_seeds[i] = mix[i]
            seeds = new_seeds

            # 4) save checkpoint this round
            if save_each_round:
                try:
                    import pickle
                    with open(f"misr_elites_n{n}_r{r+1}.pkl", "wb") as f:
                        pickle.dump(elites[:512], f)
                    print(f"   saved checkpoint: misr_elites_n{n}_r{r+1}.pkl (top {min(512,len(elites))})")
                except Exception as e:
                    print(f"   WARN: failed to save checkpoint: {e}")

        # per-size final save
        try:
            import pickle
            with open(f"misr_elites_n{n}_final.pkl", "wb") as f:
                pickle.dump(elites[:1024], f)
            print(f"Saved per-size final: misr_elites_n{n}_final.pkl")
        except Exception as e:
            print(f"WARN: failed to save per-size final: {e}")

        # 5) Curriculum lift by +lift_step
        if elites:
            n_next = n + lift_step
            lifted=[]
            for _, h, v in elites[:min(96, len(elites))]:
                h2, v2 = lift_instance(h, v, n_next, rng)
                lifted.append((h2,v2))
            seeds = lifted + seeds[:max(0, seeds_per_round - len(lifted))]
            n = n_next
        else:
            seeds = seeded_pool(n, rng, seeds_per_round)

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
    ap.add_argument("--n_start", type=int, default=14)
    ap.add_argument("--n_target", type=int, default=32)
    ap.add_argument("--rounds_per_n", type=int, default=14)
    ap.add_argument("--seeds_per_round", type=int, default=128)
    ap.add_argument("--local_time_per_seed", type=float, default=6.0)
    ap.add_argument("--elites_to_train", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--train_steps_per_round", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--lp_mode", type=str, default="open_midpoints", choices=["open_midpoints","closed_grid"])
    # two-phase schedule
    ap.add_argument("--phase_split", type=float, default=0.6)
    ap.add_argument("--alpha_lp_hi", type=float, default=0.55)
    ap.add_argument("--beta_ilp_hi", type=float, default=0.05)
    ap.add_argument("--alpha_lp_lo", type=float, default=0.20)
    ap.add_argument("--beta_ilp_lo", type=float, default=0.28)
    ap.add_argument("--grb_threads", type=int, default=0)
    ap.add_argument("--lift_step", type=int, default=3)
    ap.add_argument("--init_pkl", type=str, default="")
    ap.add_argument("--init_take", type=int, default=192)
    ap.add_argument("--save_each_round", action="store_true", default=True)

    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    elites = run_patternboost(
        seed=args.seed,
        n_start=args.n_start,
        n_target=args.n_target,
        rounds_per_n=args.rounds_per_n,
        seeds_per_round=args.seeds_per_round,
        local_time_per_seed=args.local_time_per_seed,
        elites_to_train=args.elites_to_train,
        batch_size=args.batch_size,
        train_steps_per_round=args.train_steps_per_round,
        temperature=args.temperature,
        top_p=args.top_p,
        lp_mode=args.lp_mode,
        phase_split=args.phase_split,
        alpha_lp_hi=args.alpha_lp_hi, beta_ilp_hi=args.beta_ilp_hi,
        alpha_lp_lo=args.alpha_lp_lo, beta_ilp_lo=args.beta_ilp_lo,
        grb_threads=args.grb_threads,
        lift_step=args.lift_step,
        init_pkl=args.init_pkl,
        init_take=args.init_take,
        save_each_round=args.save_each_round,
    )
    # save top elites overall
    try:
        import pickle
        with open("misr_elites_overall.pkl", "wb") as f:
            pickle.dump(elites[:512], f)
        print("Saved overall elites to misr_elites_overall.pkl")
    except Exception as e:
        print(f"WARN: failed to save overall elites: {e}")

if __name__ == "__main__":
    main()
